"""Versioned FastAPI entry point for Sparkie Mind."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .audio import AudioValidationError, validate_wav
from .runtime import InferenceRunner, build_runner
from .schemas import ErrorResponse, HealthResponse, VoiceResponse
from .settings import Settings


logger = logging.getLogger("sparkie_mind")


class AdmissionGate:
    """Bound requests before they can wait for the single inference worker."""

    def __init__(self, queue_length: int) -> None:
        self.capacity = 1 + queue_length
        self._slots = asyncio.Semaphore(self.capacity)
        self._worker = asyncio.Lock()

    async def try_acquire(self) -> bool:
        if self._slots.locked():
            return False
        await self._slots.acquire()
        return True

    async def run(self, operation: Any) -> str:
        async with self._worker:
            return await operation

    def release(self) -> None:
        self._slots.release()

    def state(self) -> dict[str, int]:
        in_flight = self.capacity - self._slots._value  # Semaphore has no public read-only counter.
        return {"capacity": self.capacity, "in_flight": in_flight, "waiting": max(0, in_flight - 1)}


class ServiceState:
    def __init__(self, settings: Settings, runner: InferenceRunner) -> None:
        self.settings = settings
        self.runner = runner
        self.gate = AdmissionGate(settings.queue_length)
        self.model_status = "loading"
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model_status == "ready"


def _error(status_code: int, code: str, message: str, request_id: UUID | None = None) -> JSONResponse:
    payload = ErrorResponse(request_id=request_id, error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="The uploaded audio exceeds the maximum size.")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json_field(value: str | None, field_name: str, default: Any) -> Any:
    if value is None or not value.strip():
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail=f"{field_name} must contain valid JSON.") from error
    if not isinstance(parsed, type(default)):
        raise HTTPException(status_code=422, detail=f"{field_name} has an invalid JSON type.")
    return parsed


def _parse_model_response(raw: str, request_id: UUID, available_tools: list[dict[str, Any]]) -> VoiceResponse:
    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[1].rsplit("\n", 1)[0]
    try:
        response = VoiceResponse.model_validate_json(candidate)
    except ValidationError as error:
        raise ValueError("The model did not return the required response schema.") from error
    if response.request_id != request_id:
        raise ValueError("The model response request_id does not match the request.")
    permitted_names = {tool.get("name") for tool in available_tools if isinstance(tool.get("name"), str)}
    if any(tool.name not in permitted_names for tool in response.tool_calls):
        raise ValueError("The model requested a tool that was not supplied by the robot.")
    return response


def _retain_debug_audio(settings: Settings, request_id: UUID, content: bytes) -> None:
    if not settings.debug_retain_audio:
        return
    settings.debug_audio_dir.mkdir(parents=True, exist_ok=True)
    path = settings.debug_audio_dir / f"{request_id}.wav"
    path.write_bytes(content)


def create_app(settings: Settings | None = None, runner: InferenceRunner | None = None) -> FastAPI:
    configured_settings = settings or Settings.from_environment()
    configured_runner = runner or build_runner(configured_settings)
    state = ServiceState(configured_settings, configured_runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await asyncio.to_thread(state.runner.load)
            state.model_status = "ready"
            logger.info("model_ready runtime=%s model=%s device=%s", configured_settings.runtime, configured_settings.model_id, state.runner.device)
        except Exception as error:
            state.model_status = "failed"
            state.load_error = str(error)
            logger.exception("model_load_failed runtime=%s model=%s", configured_settings.runtime, configured_settings.model_id)
        yield

    app = FastAPI(title="Sparkie Mind", version="1.0.0", lifespan=lifespan)
    app.state.service = state

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            ready=state.ready,
            model_status=state.model_status,
            model=configured_settings.model_id,
            device=state.runner.device,
            queue=state.gate.state(),
        )

    @app.post("/v1/voice-requests", response_model=VoiceResponse, responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def voice_request(
        request: Request,
        audio: UploadFile = File(...),
        request_id: UUID = Form(...),
        robot_id: str = Form(..., min_length=1, max_length=128),
        language: str = Form("it", min_length=2, max_length=16),
        timestamp: datetime = Form(...),
        context: str | None = Form(None),
        available_tools: str | None = Form(None),
    ) -> VoiceResponse:
        del request, timestamp  # Parsed for API validation but never logged with audio content.
        if not state.ready:
            return _error(503, "service_not_ready", "The inference service is not ready.", request_id)  # type: ignore[return-value]
        try:
            raw_audio = await _read_upload(audio, configured_settings.max_upload_bytes)
            validated_audio = validate_wav(raw_audio, audio.content_type, configured_settings.max_audio_duration_seconds)
            parsed_context = _parse_json_field(context, "context", {})
            parsed_tools = _parse_json_field(available_tools, "available_tools", [])
        except HTTPException as error:
            return _error(error.status_code, "invalid_request", str(error.detail), request_id)  # type: ignore[return-value]
        except AudioValidationError as error:
            return _error(400, "invalid_audio", str(error), request_id)  # type: ignore[return-value]
        finally:
            await audio.close()
        if not await state.gate.try_acquire():
            return _error(429, "busy", "The inference queue is full. Please retry later.", request_id)  # type: ignore[return-value]
        try:
            _retain_debug_audio(configured_settings, request_id, raw_audio)
            raw_response = await state.gate.run(
                asyncio.wait_for(
                    asyncio.to_thread(state.runner.generate, validated_audio, str(request_id), language, parsed_context, parsed_tools),
                    timeout=configured_settings.request_timeout_seconds,
                )
            )
            response = _parse_model_response(raw_response, request_id, parsed_tools)
            logger.info("voice_request_completed request_id=%s robot_id=%s duration_seconds=%.3f", request_id, robot_id, validated_audio.duration_seconds)
            return response
        except TimeoutError:
            logger.warning("voice_request_timed_out request_id=%s robot_id=%s", request_id, robot_id)
            return _error(504, "inference_timeout", "Inference exceeded the configured request timeout.", request_id)  # type: ignore[return-value]
        except ValueError as error:
            logger.warning("voice_request_invalid_model_output request_id=%s robot_id=%s", request_id, robot_id)
            return _error(502, "invalid_model_output", str(error), request_id)  # type: ignore[return-value]
        except Exception:
            logger.exception("voice_request_failed request_id=%s robot_id=%s", request_id, robot_id)
            return _error(502, "inference_failed", "The inference request failed.", request_id)  # type: ignore[return-value]
        finally:
            state.gate.release()

    return app


app = create_app()
