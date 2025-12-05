# Load testing the realtime bridge

This repository ships a K6 test plan that simulates two high-level behaviours against a deployed instance of the realtime bridge:

1. **Concurrent chat streams** that open the `/v1/realtime` WebSocket, send a user message, and wait for a streamed text delta.
2. **Feedback submissions** sent as HTTP POST requests to a configurable feedback endpoint.

## Prerequisites

* A reachable deployment URL (dev/staging) that exposes the realtime WebSocket and feedback endpoint.
* K6 installed locally (`brew install k6`, `choco install k6`, or download from https://k6.io/docs/get-started/installation/).
* Optional: an API token to pass through `Authorization: Bearer <token>`.

## Running locally

```bash
# Default targets: http://localhost:8080 and ws://localhost:8080/v1/realtime
k6 run load-tests/k6-realtime.js \
  -e BASE_URL=https://dev.example.com \
  -e WS_URL=wss://dev.example.com/v1/realtime \
  -e FEEDBACK_URL=https://dev.example.com/feedback \
  -e API_TOKEN=$DEV_API_TOKEN \
  -e CHAT_VUS=10 \
  -e CHAT_DURATION=3m \
  -e FEEDBACK_RATE=40
```

### Environment variables

| Name | Purpose | Default |
| ---- | ------- | ------- |
| `BASE_URL` | Base HTTP URL for the target deployment. | `http://localhost:8080` |
| `WS_URL` | WebSocket URL for realtime chats. Overrides the `BASE_URL`-derived value when provided. | `${BASE_URL}/v1/realtime` (with `http` → `ws` conversion) |
| `FEEDBACK_URL` | HTTP endpoint that accepts feedback submissions. | `${BASE_URL}/feedback` |
| `API_TOKEN` | Optional bearer token propagated to both WebSocket and HTTP requests. | – |
| `CHAT_VUS` | Virtual users to hold open chat sessions. | `5` |
| `CHAT_DURATION` | Duration of the chat scenario. | `2m` |
| `CHAT_TIMEOUT_MS` | Max time to wait for a first streamed token. | `12000` |
| `FEEDBACK_RATE` | Feedback submissions per minute. | `20` |
| `FEEDBACK_DURATION` | Duration of the feedback scenario. | `CHAT_DURATION` |
| `GRACEFUL_STOP` | Time to allow k6 to drain connections before stopping. | `30s` |

### Outputs and thresholds

* Results are exported to `load-tests/k6-summary.json` for CI artifact collection.
* Thresholds fail the run when:
  * WebSocket connection or response errors exceed 5%.
  * HTTP failure rate exceeds 2%.
  * p95 chat round-trip exceeds 8s.
  * p95 feedback latency exceeds 2s.

## CI integration

The `load-test.yml` workflow runs the same test plan against the dev deployment. Set the following repository or environment secrets before triggering the workflow:

* `DEV_BASE_URL` – Base HTTP URL (e.g., `https://dev.example.com`).
* `DEV_WS_URL` – WebSocket URL (e.g., `wss://dev.example.com/v1/realtime`).
* `DEV_FEEDBACK_URL` – Feedback endpoint URL.
* `DEV_API_TOKEN` – Optional bearer token.

The production deploy job is gated on successful load testing, so failed thresholds will block promotion.
