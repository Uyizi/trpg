"""Concurrency is measured inside the blocking backend, beyond caller timeouts."""
import asyncio
import importlib.util

import httpx
import pytest


def test_model_interface_exists():
    assert importlib.util.find_spec("local_trpg.model.base") is not None


def request(**updates):
    from local_trpg.model.base import GenerationRequest, Message
    return GenerationRequest(request_id="r", session_id="s", role_kind="npc",
        messages=[Message(role="user", content="你好")], **updates)


class BlockingBackend:
    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.active = 0
        self.maximum = 0
        self.calls = 0

    async def generate(self, req):
        from local_trpg.model.base import GenerationResult
        self.calls += 1
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return GenerationResult(final_text="门已打开", finish_reason="stop")
        finally:
            self.active -= 1


async def test_caller_cancellation_keeps_actual_generation_serial():
    from local_trpg.model.queue import GenerationQueue
    backend = BlockingBackend()
    queue = GenerationQueue(backend)
    first = asyncio.create_task(queue.generate(request()))
    await backend.started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(queue.generate(request()))
    await asyncio.sleep(.03)
    assert backend.calls == 1
    backend.release.set()
    assert (await second).final_text == "门已打开"
    assert backend.maximum == queue.max_active == 1


async def test_timeout_does_not_start_another_actual_generation():
    from local_trpg.model.queue import GenerationQueue
    backend = BlockingBackend()
    queue = GenerationQueue(backend)
    result = await queue.generate(request(timeout_seconds=.02))
    assert result.error == "timeout"
    second = asyncio.create_task(queue.generate(request()))
    await asyncio.sleep(.03)
    assert backend.calls == 1
    backend.release.set()
    await second
    assert backend.maximum == 1


async def test_request_that_times_out_waiting_never_generates():
    from local_trpg.model.queue import GenerationQueue
    backend = BlockingBackend()
    queue = GenerationQueue(backend)
    first = asyncio.create_task(queue.generate(request()))
    await backend.started.wait()
    result = await queue.generate(request(timeout_seconds=.02))
    assert result.error == "timeout"
    backend.release.set()
    await first
    await asyncio.sleep(.02)
    assert backend.calls == 1


async def test_http_timeout_leaves_busy_unknown_and_refuses_new_work():
    from local_trpg.model.http_backend import HTTPBackend
    from local_trpg.model.queue import GenerationQueue
    calls = []
    async def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("service may still be generating")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = HTTPBackend("http://127.0.0.1:9876", client=client)
    queue = GenerationQueue(backend)
    first = await queue.generate(request())
    second = await queue.generate(request())
    assert first.error == second.error == "backend_busy_unknown"
    assert len(calls) == 1
    await backend.close()


def test_final_output_never_exposes_unclosed_reasoning():
    from local_trpg.model.base import extract_final
    assert extract_final("<think>秘密推理</think>你好") == "你好"
    assert extract_final("<think>未完的秘密") == ""
    assert extract_final("你好") == "你好"


async def test_http_strips_reasoning_and_preserves_usage():
    from local_trpg.model.http_backend import HTTPBackend
    async def handler(req):
        return httpx.Response(200, json={"choices":[{"message":{"content":"<think>秘密</think>你好"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5}})
    backend = HTTPBackend("http://127.0.0.1:9876", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await backend.generate(request())
    assert result.final_text == "你好"
    assert result.usage == {"input_tokens":10,"output_tokens":5}
    assert backend.capabilities().supports_cancel is False
    with pytest.raises(RuntimeError, match="tokenizer"):
        backend.count_tokens(request().messages)
    await backend.close()


@pytest.mark.parametrize("mode", ["timeout", "cancel", "error"])
async def test_runner_stops_only_owned_worker_on_timeout_or_cancel(tmp_path, monkeypatch, mode):
    """A real subprocess must exit before its GPU lane can be reused."""
    import subprocess
    import sys
    from local_trpg.config import Settings
    from local_trpg.model.runner_backend import RunnerBackend
    import local_trpg.model.runner_backend as module
    helper = tmp_path / "bridge_fixture.py"
    helper.write_text('''import json, sys, time
for line in sys.stdin:
    command = json.loads(line)
    if command["op"] == "generate":
        time.sleep(10)
    print(json.dumps({"identity":{"fingerprint":"fixture"}}), flush=True)
''')
    if mode == "error":
        helper.write_text(helper.read_text().replace('time.sleep(10)', 'print(json.dumps({"error":"controlled worker error"}), flush=True); continue'))
    real_popen = subprocess.Popen
    unrelated = real_popen([sys.executable, "-c", "import time; time.sleep(30)"])
    processes = []
    def spawn(args, **kwargs):
        process = real_popen([sys.executable, "-u", str(helper)], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    backend = RunnerBackend(Settings(model_python=sys.executable, model_path=tmp_path, data_dir=tmp_path))
    try:
        assert (await backend.generate(request())).error == "backend_not_warmed"
        await backend.warmup()
        task = asyncio.create_task(backend.generate(request(timeout_seconds=.05 if mode == "timeout" else 5)))
        if mode == "cancel":
            await asyncio.sleep(.03)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "timeout":
            assert (await task).error == "timeout"
        else:
            assert (await task).error == "backend_error:controlled worker error"
        assert processes[0].poll() is not None
        assert processes[0].stdin.closed and processes[0].stdout.closed
        assert unrelated.poll() is None
        await backend.warmup()
        assert len(processes) == 2
        assert processes[1].poll() is None
    finally:
        await backend.close()
        unrelated.terminate()
        unrelated.wait(timeout=3)


async def test_http_malformed_response_fences_unknown_server_work():
    from local_trpg.model.http_backend import HTTPBackend
    calls = []
    async def handler(req):
        calls.append(req)
        return httpx.Response(502, text="upstream response lost")
    backend = HTTPBackend("http://127.0.0.1:9876", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert (await backend.generate(request())).error == "backend_busy_unknown"
    assert (await backend.generate(request())).error == "backend_busy_unknown"
    assert len(calls) == 1
    await backend.close()
