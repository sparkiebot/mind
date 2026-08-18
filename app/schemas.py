"""Public API and independently validated model-output schemas."""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseType(str, Enum):
    SPEECH = "speech"
    TOOL_CALL = "tool_call"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class VoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    type: ResponseType
    response_text: str = Field(min_length=1, max_length=2000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)

    @field_validator("tool_calls")
    @classmethod
    def response_type_matches_tools(cls, tool_calls: list[ToolCall], info: Any) -> list[ToolCall]:
        response_type = info.data.get("type")
        if response_type == ResponseType.SPEECH and tool_calls:
            raise ValueError("speech responses cannot contain tool calls")
        if response_type == ResponseType.TOOL_CALL and not tool_calls:
            raise ValueError("tool_call responses require at least one tool call")
        return tool_calls


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: UUID | None = None
    error: ErrorDetail


class TextRequest(BaseModel):
    """A robot text turn with the same metadata as a voice request."""

    request_id: UUID
    robot_id: str = Field(min_length=1, max_length=128)
    language: str = Field(default="it", min_length=2, max_length=16)
    timestamp: datetime
    text: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_is_not_whitespace(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("text must not be blank")
        return text


class HealthResponse(BaseModel):
    ready: bool
    model_status: str
    model: str
    device: str
    queue: dict[str, int]
