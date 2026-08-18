# Sparkie Mind HTTP API

## `GET /health`

Reports service readiness, model loading status, selected model, inference
device, and queue state. It remains not-ready while weights are loading.

## `POST /v1/voice-requests`

Accepts multipart form data:

- `audio`: WAV audio.
- `request_id`: robot-generated UUID.
- `robot_id`: robot identifier.
- `language`: normally `it`.
- `timestamp`: ISO 8601 timestamp.
- `context`: optional JSON context.
- `available_tools`: optional JSON list of permitted tools.

## `POST /v1/text-requests`

Accepts a JSON object with the same request metadata as the voice endpoint,
but replaces `audio` with `text`:

```json
{
  "request_id": "uuid",
  "robot_id": "sparkie-01",
  "language": "it",
  "timestamp": "2026-08-18T12:00:00Z",
  "text": "Portami in cucina",
  "context": {},
  "available_tools": []
}
```

`text` must be non-empty and no longer than `MIND_MAX_TEXT_LENGTH` (default:
4,000 characters). The text is not retained or logged by default. The endpoint
uses the same bounded inference worker and returns the same success and error
schemas as `POST /v1/voice-requests`.

Both request endpoints return this success response:

```json
{
  "request_id": "uuid",
  "type": "speech",
  "response_text": "Certo, come posso aiutarti?",
  "tool_calls": []
}
```

Errors use:

```json
{
  "request_id": "uuid",
  "error": {
    "code": "invalid_audio",
    "message": "The uploaded audio format is not supported."
  }
}
```

The server independently validates model output and tool proposals before
returning a response.

The active runtime may be `llama-server` or `stub`. With `llama-server`, this
API remains the only robot-facing endpoint: it performs the same validation and
admission control, then sends a native-audio request to a separately managed
loopback llama.cpp server.

Common errors are `invalid_audio` (400), `invalid_request` (413/422), `busy`
(429), `service_not_ready` (503), `inference_timeout` (504), and
`invalid_model_output` (502). Every application error includes the supplied
request ID when it was successfully parsed.
