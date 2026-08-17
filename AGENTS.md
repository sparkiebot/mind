# Sparkie Mind Agent Guide

## Purpose

`mind` is the standalone private-LAN inference service for the Sparkie robot.
It is intentionally separate from the ROS workspace at `../sparkie_ws` and is
expected to become its own GitHub repository.

## Architecture Boundaries

- This service receives a completed robot WAV and sends it directly to
  `google/gemma-3n-E2B-it` using a runtime that supports its native audio
  input. Do not add Whisper or another mandatory STT stage.
- The robot owns microphone processing, wake word, VAD, recording, LEDs,
  edge-playback, localization, navigation execution, motion safety, lightweight
  vision, and physical tools.
- A tool call returned by this service is only a proposed, schema-validated
  request. It must never bypass robot-side validation or claim an action has
  completed before feedback is received.

## Implementation Rules

- Use English for documentation, code comments, logs, errors, and messages.
- Keep the HTTP API versioned. The initial endpoints are `GET /health` and
  `POST /v1/voice-requests`.
- All deployment and model settings must be configurable. Never hard-code a LAN
  address or bind publicly by default.
- Use a single bounded inference worker initially. Reject overload explicitly;
  never accumulate an unbounded audio queue.
- Validate MIME declaration, RIFF/WAV structure, duration, emptiness, and
  upload size before inference. Do not retain audio or log raw user content by
  default.
- Read `docs/ARCHITECTURE.md` and `docs/API.md` before changing the service.

## Current Prerequisites

Before connecting a robot client, obtain the robot branch or files that contain
the recorder completion hook, wake-word executable, and `edge-playback`
integration. Inspect the target server OS, CPU, RAM, GPU/VRAM, CUDA support,
storage, and private LAN/firewall configuration before selecting production
device and precision defaults.
