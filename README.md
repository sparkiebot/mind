# Sparkie Mind

Sparkie Mind is Sparkie's private-LAN inference service. It accepts a completed
WAV or a text request, sends it to local Gemma 4 E2B through llama.cpp, and
returns a validated Italian response or tool proposal. Audio requests have no
separate STT/Whisper stage.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
scripts/run-llama-mind.sh
```

The script starts `llama-server` with the configured GGUF and multimodal
projector, waits for it, then starts Sparkie Mind. Press `Ctrl+C` to stop both.

## Configuration

The main settings are in [`.env.example`](.env.example). Keep
`MIND_QUEUE_LENGTH=0` until latency has been measured. Override model paths or
context length for a local run, for example:

```bash
MIND_LLAMA_CONTEXT_SIZE=8192 scripts/run-llama-mind.sh
```

Use `MIND_RUNTIME=stub` with `.venv/bin/python -m app` for API tests without
llama.cpp. Run the test suite with `.venv/bin/pytest`.

See [docs/API.md](docs/API.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
and [AGENTS.md](AGENTS.md) for API, deployment, and development details.
