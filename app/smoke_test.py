"""Load the configured runtime and validate a single local WAV without the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .audio import validate_wav
from .runtime import build_runner
from .schemas import VoiceResponse
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Sparkie Mind inference from a local WAV file.")
    parser.add_argument("audio", type=Path, help="Path to a PCM WAV file")
    arguments = parser.parse_args()
    settings = Settings.from_environment()
    if settings.runtime != "gemma":
        raise SystemExit("Set MIND_RUNTIME=gemma before running the Gemma smoke test.")
    content = arguments.audio.read_bytes()
    wav = validate_wav(content, "audio/wav", settings.max_audio_duration_seconds)
    request_id = uuid4()
    runner = build_runner(settings)
    runner.load()
    response = VoiceResponse.model_validate_json(
        runner.generate(wav, str(request_id), "it", {}, [])
    )
    if response.request_id != request_id:
        raise SystemExit("Gemma returned a response for a different request ID.")
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
