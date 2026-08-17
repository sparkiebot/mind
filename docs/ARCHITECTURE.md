# Sparkie Mind Architecture

## First Milestone

```text
Robot capture completion
  -> bounded asynchronous upload worker
  -> POST /v1/voice-requests
  -> bounded single-worker Gemma 3n inference
  -> validated structured response
  -> robot local edge-playback
```

Use Hugging Face Transformers with the official
`google/gemma-3n-E2B-it` model and its multimodal processor. Gemma receives
the decoded WAV directly through its audio pathway; no separate ASR stage is
required.

## Safety and Privacy

The service binds to a configurable interface, defaulting to loopback. A
deployment must only permit the robot's private LAN through the firewall.
Requests are bounded by queue length, upload size, duration, and inference
timeout. Uploaded audio is not retained by default. Debug retention, if
enabled, uses a dedicated directory and a unique request-ID filename.

The model receives a system prompt that requires concise Italian output,
schema-only responses, tool calls limited to those supplied by the robot, no
invented arguments, clarification for ambiguous or unsafe commands, and no
claims of completed physical actions without execution feedback.

## Configuration

At implementation time, configure at minimum: model ID/path, device, precision,
maximum generation tokens, temperature, inference timeout, maximum audio
duration, upload limit, queue length, debug retention path, bind address, and
port. Choose CUDA/BF16 only after inspecting the server hardware.

### Apple Silicon Development Profile

The development host is a 16 GB Apple M1 Mac running macOS. For direct Gemma
audio integration, configure MPS, FP16, `metal-8bit` quantization, a single
inference worker, and no waiting queue. Metal quantization is selected over
bitsandbytes because it is designed for Apple MPS. It uses the official Gemma
checkpoint and Transformers' Metal kernels; the model must still fit completely
within unified memory.

## Hardware-Free Verification

Set `MIND_RUNTIME=stub` to load the deterministic in-process runner instead of
Gemma. The stub accepts the same multipart request, performs the same WAV,
queue, and response-schema handling, and returns a fixed Italian speech
response. The test suite generates a short PCM WAV in memory, so no robot,
microphone, model weights, or GPU is needed for API verification.
