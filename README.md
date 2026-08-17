# Sparkie Mind

Sparkie Mind is the standalone private-LAN inference service for Sparkie's voice
assistant. It accepts a completed WAV from the robot, passes audio directly to
Gemma 4 E2B through local llama.cpp, and returns a validated Italian speech
response or a constrained tool proposal. It does not include a Whisper/STT
preprocessing stage.

## Install

Use Python 3.11 or later. Python dependencies support the API and the stubbed
test runtime. Native-audio inference is provided by a separately installed
`llama-server` binary.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
```

## Apple Silicon development setup

The current development machine is a 16 GB M1 Mac. Use it for compatibility,
audio-path, and latency experiments with one request at a time; do not treat it
as a proven production capacity target. llama.cpp uses Metal automatically on
macOS when `--gpu-layers 99` is supplied by the local-stack script.

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

Set `MIND_RUNTIME=llama-server` and configure values in the environment file.
See [`.env.example`](.env.example) for the complete initial set. The service
passes the WAV as native audio to a local Gemma 4 E2B llama.cpp server; it does
not create an intermediate transcript.

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

## llama.cpp audio smoke test on macOS

`llama-server` with the Gemma 4 E2B GGUF and its matching `mmproj` is a
supported external runtime. Sparkie Mind continues to own its API validation,
bounded admission queue, timeout, and output-schema validation; llama.cpp owns
only the native-audio model inference. Its Metal build works on the M1
development machine. Start it on loopback with `--jinja`, `--reasoning off`,
and the model paths configured for the local checkout:

```bash
llama-server \
  --model models/llama.cpp/gemma4-e2b/google-gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj models/llama.cpp/gemma4-e2b/mmproj-BF16.gguf \
  --jinja --reasoning off --gpu-layers 99 --ctx-size 2048 \
  --host 127.0.0.1 --port 8080
```

Record a non-empty PCM WAV with the macOS microphone. Give Terminal microphone
permission first, discover the audio-device index, then replace `0` below with
that index:

```bash
ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -i ":0" -t 5 -ar 48000 -ac 1 -c:a pcm_s16le tmp-poc/mic.wav
```

The browser test page described below is the recommended way to record and send
audio through the complete Sparkie Mind API. A `400` containing `Unable to read
WAV audio file from buffer` indicates an invalid or empty recording; verify the
file has non-zero frames before attributing the error to Metal or the server.

To run the Sparkie Mind API through this runtime, configure the API in a second
terminal after `llama-server` is ready:

```bash
MIND_RUNTIME=llama-server \
MIND_LLAMA_SERVER_URL=http://127.0.0.1:8080 \
MIND_LLAMA_SERVER_MODEL=gemma4-e2b \
MIND_QUEUE_LENGTH=0 \
.venv/bin/python -m app
```

The shared system prompt makes Sparkie answer the user directly, not transcribe
the input. Ordinary responses are limited to one short Italian sentence unless
the user asks for an explanation.

### Start both local services

Use the development-stack script to start `llama-server`, wait for it to become
ready, and then start Sparkie Mind with the correct runtime configuration:

```bash
scripts/run-llama-mind.sh
```

Press `Ctrl+C` once to stop both processes. llama.cpp output is written to
`logs/llama-server.log`; Sparkie Mind logs remain in the terminal. Override
model paths or tuning values for one run with environment variables, for
example:

```bash
MIND_LLAMA_CONTEXT_SIZE=8192 scripts/run-llama-mind.sh
```

### Browser microphone test

With Sparkie Mind running, open [http://127.0.0.1:8088/test](http://127.0.0.1:8088/test).
The page records microphone input, encodes a PCM WAV locally, offers playback,
and sends it to the normal `/v1/voice-requests` endpoint. Browser microphone
access works over loopback HTTP during development. Do not use an unencrypted
private-LAN URL for this page: browsers require HTTPS for microphone access on
non-loopback origins.
