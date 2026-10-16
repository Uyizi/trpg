"""Optional local chat/completions adapter; uncertain requests remain fenced."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from .base import GenerationRequest, GenerationResult, Message, ModelCapabilities, extract_final


class HTTPBackend:
    def __init__(self, base_url: str, *, model: str = "local-model", client: httpx.AsyncClient | None = None,
                 token_counter: Callable[[list[Message]], int] | None = None):
        if urlparse(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a configured local model service is supported")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.AsyncClient(trust_env=False)
        self.token_counter = token_counter
        self.busy_unknown = False
        self._lock = asyncio.Lock()

    def capabilities(self):
        return ModelCapabilities(model_identity=f"http:{self.model}")

    def count_tokens(self, messages):
        if self.token_counter is None:
            raise RuntimeError("HTTP backend needs a verified tokenizer; estimates are not accepted")
        return self.token_counter(messages)

    async def warmup(self):
        return self.capabilities()

    async def generate(self, request: GenerationRequest):
        start = time.monotonic()
        async with self._lock:
            if self.busy_unknown:
                return GenerationResult(error="backend_busy_unknown")
            try:
                response = await self.client.post(self.base_url + "/v1/chat/completions",
                    json={"model": self.model, "messages": [m.model_dump() for m in request.messages],
                          "max_tokens": request.max_output_tokens, "temperature": request.temperature, "stream": False},
                    timeout=request.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return GenerationResult(final_text=extract_final(choice["message"]["content"] or ""),
                    finish_reason=choice.get("finish_reason") or "unknown",
                    usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
                    elapsed_ms=(time.monotonic() - start) * 1000)
            except (httpx.TimeoutException, httpx.TransportError, asyncio.CancelledError):
                self.busy_unknown = True
                return GenerationResult(error="backend_busy_unknown", elapsed_ms=(time.monotonic() - start) * 1000)
            except (httpx.HTTPStatusError, KeyError, ValueError, TypeError):
                # Even a gateway error cannot certify that upstream work stopped.
                self.busy_unknown = True
                return GenerationResult(error="backend_busy_unknown", elapsed_ms=(time.monotonic() - start) * 1000)

    async def close(self):
        await self.client.aclose()
