from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
from uuid import uuid4
import wave

from fastapi.testclient import TestClient

from app.__main__ import HealthRequestFilter
from app.main import AdmissionGate, create_app
from app.runtime import InferenceRunner, LlamaServerRunner, StubRunner
from app.settings import Settings


def wav_bytes(seconds: float = 0.1, sample_rate: int = 16_000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buffer.getvalue()


def client(*, max_duration: float = 30.0) -> TestClient:
    settings = Settings(runtime="stub", max_audio_duration_seconds=max_duration)
    return TestClient(create_app(settings=settings, runner=StubRunner()))


def request_parts(audio: bytes, content_type: str = "audio/wav") -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    return (
        {
            "request_id": str(uuid4()),
            "robot_id": "sparkie-test",
            "language": "it",
            "timestamp": "2026-08-17T12:00:00Z",
            "context": json.dumps({"battery_percent": 80}),
            "available_tools": json.dumps([]),
        },
        {"audio": ("request.wav", audio, content_type)},
    )


def text_request_payload(text: str = "Portami in cucina") -> dict[str, object]:
    return {
        "request_id": str(uuid4()),
        "robot_id": "sparkie-test",
        "language": "it",
        "timestamp": "2026-08-17T12:00:00Z",
        "text": text,
        "context": {"battery_percent": 80},
        "available_tools": [],
    }


def test_health_reports_stub_readiness() -> None:
    with client() as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "model_status": "ready",
        "model": "gemma4-e2b",
        "device": "stub",
        "queue": {"capacity": 1, "in_flight": 0, "waiting": 0},
    }


def test_health_access_log_filter_suppresses_only_health_requests() -> None:
    log_filter = HealthRequestFilter()
    health_record = logging.makeLogRecord(
        {"name": "uvicorn.access", "msg": '127.0.0.1 - "GET /health HTTP/1.1" 200'}
    )
    voice_record = logging.makeLogRecord(
        {"name": "uvicorn.access", "msg": '127.0.0.1 - "POST /v1/voice-requests HTTP/1.1" 200'}
    )

    assert not log_filter.filter(health_record)
    assert log_filter.filter(voice_record)


def test_stub_accepts_valid_wav_without_robot() -> None:
    data, files = request_parts(wav_bytes())
    with client() as test_client:
        response = test_client.post("/v1/voice-requests", data=data, files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == data["request_id"]
    assert payload["type"] == "speech"
    assert payload["response_text"] == "Certo, come posso aiutarti?"
    assert payload["tool_calls"] == []


def test_stub_accepts_text_request_without_robot() -> None:
    payload = text_request_payload()
    with client() as test_client:
        response = test_client.post("/v1/text-requests", json=payload)
    assert response.status_code == 200
    assert response.json()["request_id"] == payload["request_id"]
    assert response.json()["type"] == "speech"


def test_text_request_rejects_over_limit_without_logging_content() -> None:
    payload = text_request_payload("x" * 11)
    settings = Settings(runtime="stub", max_text_length=10)
    with TestClient(create_app(settings=settings, runner=StubRunner())) as test_client:
        response = test_client.post("/v1/text-requests", json=payload)
    assert response.status_code == 413
    assert response.json()["request_id"] == payload["request_id"]
    assert response.json()["error"]["code"] == "invalid_request"


def test_text_request_rejects_blank_text() -> None:
    payload = text_request_payload("   ")
    with client() as test_client:
        response = test_client.post("/v1/text-requests", json=payload)
    assert response.status_code == 422


def test_rejects_non_wav_and_preserves_request_id() -> None:
    data, files = request_parts(b"not a wav", "audio/mpeg")
    with client() as test_client:
        response = test_client.post("/v1/voice-requests", data=data, files=files)
    assert response.status_code == 400
    assert response.json()["request_id"] == data["request_id"]
    assert response.json()["error"]["code"] == "invalid_audio"


def test_rejects_over_duration_wav() -> None:
    data, files = request_parts(wav_bytes(seconds=0.2))
    with client(max_duration=0.1) as test_client:
        response = test_client.post("/v1/voice-requests", data=data, files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio"


class UnauthorizedToolRunner(InferenceRunner):
    device = "test"

    def load(self) -> None:
        return None

    def generate(self, audio, request_id, language, context, tools) -> str:
        return self._response(request_id)

    def generate_text(self, text, request_id, language, context, tools) -> str:
        return self._response(request_id)

    @staticmethod
    def _response(request_id: str) -> str:
        return json.dumps(
            {
                "request_id": request_id,
                "type": "tool_call",
                "response_text": "Procedo.",
                "tool_calls": [{"name": "invented_tool", "arguments": {}}],
            }
        )


def test_rejects_model_tool_not_supplied_by_robot() -> None:
    data, files = request_parts(wav_bytes())
    settings = Settings(runtime="stub")
    with TestClient(create_app(settings=settings, runner=UnauthorizedToolRunner())) as test_client:
        response = test_client.post("/v1/voice-requests", data=data, files=files)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


def test_admission_gate_has_no_unbounded_backlog() -> None:
    async def check() -> None:
        gate = AdmissionGate(queue_length=0)
        assert await gate.try_acquire()
        assert not await gate.try_acquire()
        gate.release()
        assert await gate.try_acquire()
        gate.release()

    import asyncio

    asyncio.run(check())


def test_llama_runtime_builds_a_direct_response_request() -> None:
    audio = wav_bytes()
    settings = Settings(runtime="llama-server", llama_server_model="gemma4-e2b")
    payload = LlamaServerRunner(settings).build_audio_chat_request(
        audio,
        "it",
        {"battery_percent": 80},
        [{"name": "navigate", "description": "Move Sparkie."}],
    )

    assert settings.selected_model == "gemma4-e2b"
    assert payload["model"] == "gemma4-e2b"
    assert "do not transcribe" in payload["messages"][0]["content"]
    assert "request_id" not in payload["response_format"]["schema"]["properties"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["schema"]["additionalProperties"] is False
    assert payload["max_tokens"] == settings.max_generated_tokens
    content = payload["messages"][1]["content"]
    assert base64.b64decode(content[0]["input_audio"]["data"]) == audio
    assert "Request ID" not in content[1]["text"]
    assert "navigate" in content[1]["text"]


def test_llama_runtime_builds_text_request() -> None:
    settings = Settings(runtime="llama-server", llama_server_model="gemma4-e2b")
    payload = LlamaServerRunner(settings).build_text_chat_request("Portami in cucina", "it", {}, [])

    assert payload["messages"][1]["content"].startswith("User request: Portami in cucina")
    assert "supplied text" in payload["messages"][1]["content"]
