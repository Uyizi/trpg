"""Run one real NPC line and one real JSON response using existing local weights."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_trpg.model.base import GenerationRequest, Message
from local_trpg.model.queue import GenerationQueue
from local_trpg.model.runner_backend import RunnerBackend


async def main():
    backend = RunnerBackend()
    queue = GenerationQueue(backend)
    started = time.monotonic()
    report = {"kind": "real_model_smoke", "success": False, "outputs": []}
    try:
        report["runtime"] = await queue.warmup()
        report["cold_load_ms"] = round((time.monotonic() - started) * 1000)
        report["capabilities"] = queue.capabilities().model_dump()
        for kind, prompt in (("npc", "你是空间站值班工程师林岚。玩家问：冷却系统怎么了？用中文说一句简短台词，不超过30个汉字。"),
                             ("json", '只输出一个JSON对象，不要代码块或解释。对象必须是 {"speech":"我去检查阀门。","action":"inspect"}。')):
            messages = [Message(role="system", content="你是文字冒险中的角色。遵守输出格式，不输出思考过程。"), Message(role="user", content=prompt)]
            count = queue.count_tokens(messages)
            result = await queue.generate(GenerationRequest(request_id=f"smoke-{kind}", session_id="smoke",
                role_kind="npc", messages=messages, max_output_tokens=128, temperature=0,
                response_schema={"type": "object"} if kind == "json" else None))
            if result.error or not result.final_text or result.finish_reason != "stop":
                raise RuntimeError(f"Smoke {kind} failed: {result.model_dump()}")
            if result.usage["input_tokens"] != count:
                raise RuntimeError("Token-only and real inference tokenizer counts disagree")
            if kind == "json":
                value = json.loads(result.final_text)
                if value != {"speech": "我去检查阀门。", "action": "inspect"}:
                    raise RuntimeError("Generated JSON did not match the requested object")
            report["outputs"].append({"kind": kind, "counted_input_tokens": count, **result.model_dump(), "stats": backend.last_stats})
        report["max_actual_queue_active"] = queue.max_active
        report["success"] = True
    finally:
        await queue.close()
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
