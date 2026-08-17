"""Strict in-memory WAV validation. No request audio is persisted here."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import wave


ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WavAudio:
    content: bytes
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float


def validate_wav(content: bytes, content_type: str | None, max_duration_seconds: float) -> WavAudio:
    mime_type = (content_type or "").split(";", 1)[0].strip().lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise AudioValidationError("The uploaded audio MIME type is not supported.")
    if not content:
        raise AudioValidationError("The uploaded audio is empty.")
    try:
        with wave.open(BytesIO(content), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise AudioValidationError("Only uncompressed PCM WAV audio is supported.")
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.getnframes()
            sample_width = wav.getsampwidth()
    except wave.Error as error:
        raise AudioValidationError("The uploaded file is not a valid WAV file.") from error
    if sample_rate <= 0 or channels not in {1, 2} or sample_width not in {1, 2, 3, 4} or frames <= 0:
        raise AudioValidationError("The uploaded WAV has unsupported audio parameters.")
    duration_seconds = frames / sample_rate
    if duration_seconds > max_duration_seconds:
        raise AudioValidationError("The uploaded audio exceeds the maximum duration.")
    return WavAudio(content, sample_rate, channels, frames, duration_seconds)
