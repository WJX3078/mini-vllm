"""OpenAI-compatible request/response models (pydantic v2).

Only engine-supported parameters are accepted. Known OpenAI parameters the
engine does NOT implement are declared in UNSUPPORTED_PARAMS and rejected
with an explicit 400 (never silently ignored).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

UNSUPPORTED_PARAMS = {
    "logprobs": "log probabilities are not implemented",
    "top_logprobs": "log probabilities are not implemented",
    "best_of": "best_of is not implemented",
    "echo": "echoing the prompt is not implemented",
    "presence_penalty": "penalties are not implemented",
    "frequency_penalty": "penalties are not implemented",
    "logit_bias": "logit bias is not implemented",
    "tools": "tool calling is not implemented",
    "tool_choice": "tool calling is not implemented",
    "functions": "function calling is not implemented",
    "response_format": "structured output is not implemented",
}


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    code: int


def error_body(message: str, err_type: str, code: int) -> dict:
    return {"error": {"message": message, "type": err_type, "code": code}}


class CompletionRequest(BaseModel):
    # extra="allow": unknown OpenAI parameters land in model_extra and are
    # explicitly rejected by check_unsupported() with a 400
    model_config = ConfigDict(extra="allow")

    model: str = ""
    prompt: str | list[str] | list[int] = ""
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stream: bool = False
    seed: int | None = None
    stop: list[str] | str | None = None
    ignore_eos: bool = False

    def normalized_stop(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stream: bool = False
    seed: int | None = None
    stop: list[str] | str | None = None
    ignore_eos: bool = False

    def normalized_stop(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)
