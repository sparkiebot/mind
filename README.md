# Sparkie Mind

Sparkie Mind is the standalone private-LAN inference service for Sparkie's voice
assistant. It accepts a completed WAV from the robot, passes audio directly to
Gemma 3n, and returns a validated Italian speech response or a constrained tool
proposal. It does not include a Whisper/STT preprocessing stage.

## Install

Use Python 3.11 or later. The default dependencies support the API and the
stubbed test runtime. Gemma dependencies are optional so a developer can test
the service without model weights or a GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
```

For real inference, first accept the Gemma license on Hugging Face and configure
an access token if required. Install a CUDA-compatible PyTorch build for the
server, then install the production extra:

```bash
.venv/bin/pip install -e '.[gemma]'
```

## Apple Silicon development setup

The current development machine is a 16 GB M1 Mac. Use it for compatibility,
audio-path, and latency experiments with one request at a time; do not treat it
as a proven production capacity target. The Metal 8-bit path keeps the official
Gemma checkpoint and quantizes eligible weights on the MPS device.

```bash
.venv/bin/python -m app.preflight
MIND_RUNTIME=gemma \
MIND_DEVICE=mps \
MIND_PRECISION=float16 \
MIND_QUANTIZATION=metal-8bit \
MIND_QUEUE_LENGTH=0 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/python -m app.smoke_test /absolute/path/to/italian-request.wav
```

The first run downloads the official model, Metal kernels, and creates the
quantized representation. Keep sufficient free disk space and close memory
heavy applications. `PYTORCH_ENABLE_MPS_FALLBACK=1` makes unsupported MPS
operations run on CPU rather than fail; measure latency carefully when it is
needed.

## Run without a robot or model

The stub is deterministic and performs the same multipart, WAV, schema, and
queue admission handling as production:

```bash
MIND_RUNTIME=stub .venv/bin/python -m app
.venv/bin/pytest
```

The default bind address is `127.0.0.1:8088`. Check readiness with:

```bash
curl http://127.0.0.1:8088/health
```

## Production configuration

Set `MIND_RUNTIME=gemma` and configure values in the environment file. See
[`.env.example`](.env.example) for the complete initial set. The service uses
`google/gemma-3n-E2B-it` through Transformers' audio-capable processor; it
decodes the WAV only to feed the native Gemma audio input, not to generate a
transcript.

Start with one worker and `MIND_QUEUE_LENGTH=0`. Increase the queue only after
measuring server GPU/VRAM and latency. The health endpoint exposes the current
bounded admission state.

## Deployment

`systemd/sparkie-mind.service` is an example unit. Install the repository at
`/opt/sparkie-mind`, create the `sparkie-mind` service account, and place
configuration in `/etc/sparkie-mind/environment`. The unit's writable data path
is `/var/lib/sparkie-mind`; set `MIND_DEBUG_AUDIO_DIR` there if debug retention
is explicitly enabled.

Do not bind to `0.0.0.0` unless a firewall permits the configured TCP port only
from the robot's trusted private subnet. Do not expose this service publicly.

The API contract and design constraints are in [docs/API.md](docs/API.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
