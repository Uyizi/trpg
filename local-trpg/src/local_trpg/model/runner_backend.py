"""Bridge to the unchanged Qwen runner; owns only its own worker processes."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time

from ..config import Settings
from .base import GenerationRequest, GenerationResult, Message, ModelCapabilities, extract_final


class _Bridge:
    def __init__(self, config: Settings, *, tokenizer_only=False):
        self.config = config
        self.tokenizer_only = tokenizer_only
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.log = None
        self.closed = False

    def _start(self):
        if self.closed:
            raise RuntimeError("backend_closed")
        if self.process is not None and self.process.poll() is None:
            return
        if not self.config.model_python.is_file():
            raise RuntimeError("Configured existing model Python is missing")
        if not self.config.model_path.is_dir():
            raise RuntimeError("Configured local model directory is missing")
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONDONTWRITEBYTECODE="1", LOCAL_TRPG_PARENT_PID=str(os.getpid()))
        # No proxy use or package/model downloads occur in inference workers.
        for key in tuple(env):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                env.pop(key)
        env["NO_PROXY"] = "*"
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        if self.log is not None:
            self.log.close()
        self.log = (self.config.data_dir / "model-worker.log").open("a", encoding="utf-8")
        args = [str(self.config.model_python), "-u", str(Path(__file__).with_name("worker.py")),
            str(self.config.model_project), str(self.config.model_path)]
        if self.tokenizer_only:
            args.append("--tokenizer-only")
        self.process = subprocess.Popen(args, cwd=self.config.model_project, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            bufsize=1, start_new_session=True)

    def rpc(self, payload: dict, timeout: float, cancelled: threading.Event | None = None):
        with self.lock:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("not_dispatched")
            self._start()
            process = self.process
            try:
                if cancelled is not None and cancelled.is_set():
                    self.stop()
                    raise RuntimeError("not_dispatched")
                process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], timeout)
                if not ready:
                    self.stop()
                    raise TimeoutError("worker_timeout_stopped")
                line = process.stdout.readline()
                if not line:
                    self.stop()
                    raise RuntimeError("model_worker_exited")
                result = json.loads(line)
                if result.get("error"):
                    self.stop()
                    raise RuntimeError(result["error"])
                return result
            except TimeoutError:
                raise
            except (BrokenPipeError, OSError, ValueError):
                self.stop()
                raise RuntimeError("model_worker_protocol_error")
            finally:
                if process.poll() is not None:
                    process.stdin.close()
                    process.stdout.close()

    def stop(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            # Reads in the RPC thread must finish before their streams close.
            self.process = None

    def close(self):
        self.closed = True
        process = self.process
        self.stop()
        with self.lock:
            # Also catches a process whose Popen was in flight when close began.
            late_process = self.process
            self.stop()
            for retired in (process, late_process):
                if retired is not None:
                    retired.stdin.close()
                    retired.stdout.close()
            if self.log is not None:
                self.log.close()
                self.log = None


class RunnerBackend:
    def __init__(self, config: Settings | None = None):
        self.config = config or Settings()
        self._bridge = _Bridge(self.config)
        self._token_bridge = _Bridge(self.config, tokenizer_only=True)
        self._lock = asyncio.Lock()
        self._counts: OrderedDict[str, int] = OrderedDict()
        self._ready = False
        self.runtime_info: dict = {}
        self.last_stats: dict = {}

    async def _call(self, payload: dict, timeout: float):
        cancelled = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self._bridge.rpc, payload, timeout, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.to_thread(self._bridge.stop)
            # to_thread cancellation cannot stop its thread. Join the actual RPC
            # before releasing the backend lock or reporting cancellation.
            await asyncio.gather(task, return_exceptions=True)
            raise

    @property
    def available(self):
        process = self._bridge.process
        if self._ready and (process is None or process.poll() is not None):
            self._ready = False
        return self._ready

    def capabilities(self) -> ModelCapabilities:
        identity = self.runtime_info.get("identity", {}).get("fingerprint", "not_loaded")
        return ModelCapabilities(structured_output_mode=self.runtime_info.get('structured_output_mode','none'), supports_thinking_control=True,
            supports_cancel=True, supports_seed=False, max_context_tokens=6144,
            model_identity=f"qwen35-custom-nf4:{identity}")

    def count_tokens(self, messages: list[Message]) -> int:
        content = [message.model_dump() for message in messages]
        key = json.dumps(content, ensure_ascii=False, sort_keys=True)
        if key not in self._counts:
            self._counts[key] = self._token_bridge.rpc({"op": "count", "messages": content}, 60)["count"]
            if len(self._counts) > 512:
                self._counts.popitem(last=False)
        return self._counts[key]

    async def warmup(self):
        async with self._lock:
            if not self._ready:
                try:
                    self.runtime_info = await self._call({"op": "warmup"}, self.config.warmup_timeout_seconds)
                except BaseException:
                    await asyncio.to_thread(self._bridge.stop)
                    raise
                self._ready = True
            return self.runtime_info

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.monotonic()
        async with self._lock:
            if not self._ready:
                return GenerationResult(error="backend_not_warmed")
            if request.thinking_policy == "enabled":
                return GenerationResult(error="thinking_enabled_not_supported_by_fixed_profile")
            try:
                result = await self._call({"op": "generate", "request": request.model_dump(mode="json")},
                    min(request.timeout_seconds, self.config.generation_timeout_seconds))
                self.last_stats = result.get("stats", {})
                return GenerationResult(final_text=extract_final(result["text"]), finish_reason=result["finish_reason"],
                    usage={**result["usage"], "peak_torch_memory_mib": int(self.last_stats.get("peak_torch_memory_mib",0))}, elapsed_ms=(time.monotonic() - start) * 1000)
            except TimeoutError:
                self._ready = False
                return GenerationResult(error="timeout", finish_reason="timeout", elapsed_ms=(time.monotonic() - start) * 1000)
            except asyncio.CancelledError:
                await asyncio.to_thread(self._bridge.stop)
                self._ready = False
                raise
            except Exception as exc:
                await asyncio.to_thread(self._bridge.stop)
                self._ready = False
                return GenerationResult(error=f"backend_error:{exc}", elapsed_ms=(time.monotonic() - start) * 1000)

    async def close(self):
        await asyncio.gather(asyncio.to_thread(self._bridge.close), asyncio.to_thread(self._token_bridge.close))
        self._ready = False
