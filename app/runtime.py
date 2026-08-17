"""Inference backends. The stub is for tests; llama.cpp receives audio directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .audio import WavAudio
from .settings import Settings


SYSTEM_PROMPT = """You are Sparkie, a helpful voice assistant robot speaking directly with the user.
The attached audio is the user's turn. Understand its meaning and respond to the user; do not transcribe, quote, or
repeat the audio unless the user explicitly asks for a transcription. Respond in natural Italian unless the user
clearly requests another language. Be concise: normally use one short sentence and never use more than two short
sentences unless the user explicitly asks for an explanation or detail. Do not add greetings, filler, or a recap.

Never state or imply that a physical action has completed. When an action is requested, emit only a tool call that is
present in the supplied tool list and use response_text only to briefly acknowledge or clarify it. Never invent tool
names, arguments, observations, or execution results. Ask a short clarification question when the request is
ambiguous, unsafe, or cannot be fulfilled with the supplied tools.

Return only valid JSON matching this schema: {request_id: string UUID, type: 'speech' or 'tool_call', response_text:
string, tool_calls: [{name: string, arguments: object}]}. A speech response must have an empty tool_calls list. A
tool_call response must contain at least one permitted tool call."""

VOICE_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["request_id", "type", "response_text", "tool_calls"],
    "properties": {
        "request_id": {"type": "string"},
        "type": {"type": "string", "enum": ["speech", "tool_call"]},
        "response_text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
            },
        },
    },
}


class InferenceRunner(ABC):
    device: str

    @abstractmethod
    def load(self) -> None:
        """Load the backend synchronously before the server becomes ready."""

    @abstractmethod
    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        """Return the model's raw structured response."""


class StubRunner(InferenceRunner):
    """Deterministic test backend that never inspects or retains request audio."""

    device = "stub"

    def load(self) -> None:
        return None

    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "request_id": request_id,
                "type": "speech",
                "response_text": "Certo, come posso aiutarti?",
                "tool_calls": [],
            }
        )


def request_instruction(request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"Request ID: {request_id}",
            f"Language hint: {language}",
            f"Robot context: {json.dumps(context, ensure_ascii=False)}",
            f"Available tools: {json.dumps(tools, ensure_ascii=False)}",
            "Use the attached audio as the user's request and return the required JSON only.",
        ]
    )


class LlamaServerRunner(InferenceRunner):
    """Adapter for a separately managed local llama.cpp multimodal server."""

    device = "llama-server"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> None:
        request = Request(f"{self.settings.llama_server_url}/health", method="GET")
        try:
            with urlopen(request, timeout=min(10.0, self.settings.request_timeout_seconds)):
                return None
        except HTTPError as error:
            raise RuntimeError(f"llama-server health check returned HTTP {error.code}.") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach llama-server: {error.reason}") from error

    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        payload = self.build_chat_request(audio.content, request_id, language, context, tools)
        request = Request(
            f"{self.settings.llama_server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            raise RuntimeError(f"llama-server returned HTTP {error.code}.") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach llama-server: {error.reason}") from error
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("llama-server returned an unexpected chat response.") from error

    def build_chat_request(
        self,
        audio: bytes,
        request_id: str,
        language: str,
        context: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model": self.settings.llama_server_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio).decode("ascii"),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": request_instruction(request_id, language, context, tools)},
                    ],
                },
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_generated_tokens,
            "response_format": {"type": "json_schema", "schema": VOICE_RESPONSE_JSON_SCHEMA},
        }


def build_runner(settings: Settings) -> InferenceRunner:
    if settings.runtime == "stub":
        return StubRunner()
    return LlamaServerRunner(settings)
