"""Configuration loaded exclusively from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

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
    runtime: str = "llama-server"
    llama_server_url: str = "http://127.0.0.1:8080"
    llama_server_model: str = "gemma4-e2b"
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

    @property
    def selected_model(self) -> str:
        return self.llama_server_model

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        defaults = cls()
        runtime = os.getenv("MIND_RUNTIME", defaults.runtime).lower()
        if runtime not in {"llama-server", "stub"}:
            raise ValueError("MIND_RUNTIME must be 'llama-server' or 'stub'")
        llama_server_url = os.getenv("MIND_LLAMA_SERVER_URL", defaults.llama_server_url).rstrip("/")
        parsed_llama_server_url = urlparse(llama_server_url)
        if parsed_llama_server_url.scheme not in {"http", "https"} or not parsed_llama_server_url.netloc:
            raise ValueError("MIND_LLAMA_SERVER_URL must be an absolute HTTP(S) URL.")
        return cls(
            runtime=runtime,
            llama_server_url=llama_server_url,
            llama_server_model=os.getenv("MIND_LLAMA_SERVER_MODEL", defaults.llama_server_model),
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
