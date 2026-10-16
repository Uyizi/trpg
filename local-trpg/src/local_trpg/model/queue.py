"""Keep the physical backend occupied even after its caller has gone away."""
from __future__ import annotations

import asyncio
import time

from .base import GenerationRequest, GenerationResult, Message, ModelBackend


class GenerationQueue:
    def __init__(self, backend: ModelBackend):
        self.backend = backend
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self.active = 0
        self.max_active = 0
        self.completed = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        self._closed = False

    def capabilities(self):
        return self.backend.capabilities()

    def count_tokens(self, messages: list[Message]) -> int:
        return self.backend.count_tokens(messages)

    async def warmup(self):
        async with self._lock:
            method = getattr(self.backend, "warmup", None)
            return await method() if method else None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._closed:
            return GenerationResult(error="backend_closed")
        start = time.monotonic()
        abandoned = asyncio.Event()
        task = asyncio.create_task(self._execute(request, start + request.timeout_seconds, abandoned))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        try:
            # Shielding is essential: cancellation of an await is not proof that
            # a thread, subprocess, or remote GPU has stopped generating.
            return await asyncio.wait_for(asyncio.shield(task), request.timeout_seconds)
        except TimeoutError:
            abandoned.set()
            return GenerationResult(finish_reason="timeout", error="timeout", elapsed_ms=(time.monotonic() - start) * 1000)
        except asyncio.CancelledError:
            abandoned.set()
            raise

    async def _execute(self, request, deadline, abandoned):
        async with self._lock:
            if abandoned.is_set() or time.monotonic() >= deadline or self._closed:
                return GenerationResult(finish_reason="cancelled", error="not_dispatched")
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                result = await self.backend.generate(request)
                self.completed += 1
                for key in self.usage:
                    self.usage[key] += result.usage.get(key, 0)
                return result
            except Exception as exc:
                return GenerationResult(error=f"backend_error:{type(exc).__name__}")
            finally:
                self.active -= 1

    async def close(self):
        self._closed = True
        method = getattr(self.backend, "close", None)
        if method:
            await method()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
