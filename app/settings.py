"""Configuration loaded exclusively from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def _integer(name: str, default: int, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _decimal(name: str, default: float, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    runtime: str = "gemma"
    model_id: str = "google/gemma-3n-E2B-it"
    device: str = "auto"
    precision: str = "bfloat16"
    bind_host: str = "127.0.0.1"
    port: int = 8088
    max_generated_tokens: int = 256
    temperature: float = 0.2
    request_timeout_seconds: float = 45.0
    max_audio_duration_seconds: float = 30.0
    max_upload_bytes: int = 5 * 1024 * 1024
    queue_length: int = 0
    debug_retain_audio: bool = False
    debug_audio_dir: Path = Path("debug-audio")

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        defaults = cls()
        runtime = os.getenv("MIND_RUNTIME", defaults.runtime).lower()
        if runtime not in {"gemma", "stub"}:
            raise ValueError("MIND_RUNTIME must be either 'gemma' or 'stub'")
        return cls(
            runtime=runtime,
            model_id=os.getenv("MIND_MODEL_ID", defaults.model_id),
            device=os.getenv("MIND_DEVICE", defaults.device),
            precision=os.getenv("MIND_PRECISION", defaults.precision),
            bind_host=os.getenv("MIND_BIND_HOST", defaults.bind_host),
            port=_integer("MIND_PORT", defaults.port, 1),
            max_generated_tokens=_integer("MIND_MAX_GENERATED_TOKENS", defaults.max_generated_tokens, 1),
            temperature=_decimal("MIND_TEMPERATURE", defaults.temperature),
            request_timeout_seconds=_decimal("MIND_REQUEST_TIMEOUT_SECONDS", defaults.request_timeout_seconds, 0.1),
            max_audio_duration_seconds=_decimal("MIND_MAX_AUDIO_DURATION_SECONDS", defaults.max_audio_duration_seconds, 0.1),
            max_upload_bytes=_integer("MIND_MAX_UPLOAD_BYTES", defaults.max_upload_bytes, 1),
            queue_length=_integer("MIND_QUEUE_LENGTH", defaults.queue_length),
            debug_retain_audio=os.getenv("MIND_DEBUG_RETAIN_AUDIO", "false").lower() in {"1", "true", "yes"},
            debug_audio_dir=Path(os.getenv("MIND_DEBUG_AUDIO_DIR", "debug-audio")),
        )
