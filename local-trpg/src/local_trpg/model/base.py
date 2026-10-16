"""Model protocol independent of the game and its private state."""
from __future__ import annotations

import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str


class ModelCapabilities(BaseModel):
    structured_output_mode: Literal["none", "json", "json_schema"] = "none"
    supports_thinking_control: bool = False
    supports_cancel: bool = False
    supports_seed: bool = False
    max_context_tokens: int = 6144
    model_identity: str = "unknown"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    request_id: str
    session_id: str
    role_kind: str
    messages: list[Message] = Field(min_length=1)
    max_output_tokens: int = Field(default=384, ge=1, le=6144)
    temperature: float = Field(default=.2, ge=0, le=2)
    response_schema: dict[str, Any] | None = None
    prefix_message_count: int = Field(default=0, ge=0)
    cache_namespace: str | None = Field(default=None, max_length=512)
    thinking_policy: Literal["disabled", "enabled", "auto"] = "disabled"
    timeout_seconds: float = Field(default=120, gt=0, le=120)


class GenerationResult(BaseModel):
    final_text: str = ""
    finish_reason: str = "error"
    usage: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: float = 0
    error: str | None = None


class ModelBackend(Protocol):
    def capabilities(self) -> ModelCapabilities: ...
    def count_tokens(self, messages: list[Message]) -> int: ...
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


def extract_final(text: str) -> str:
    """Discard reasoning, including any unfinished reasoning tail."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    # A backend may omit the opening tag because it came from its template.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()
