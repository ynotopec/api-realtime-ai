# Remaining differences from the latest OpenAI Realtime protocol

The WebSocket bridge now emits `response.output_item.added`, `response.output_text.delta` / `done`, `response.audio.delta` / `done`, and completes each turn with `response.done`.

The points below summarise what still diverges from the reference implementation exposed at `https://api.openai.com/v1/realtime`.

| Area | Implementation here | Current OpenAI spec | Notes |
| --- | --- | --- | --- |
| Text streaming cadence | Entire reply emitted in a single `response.output_text.delta` | Token-sized deltas (`response.output_text.delta`) | Current behaviour is technically valid but differs from streaming-first examples. |
| Event naming | `response.output_item.added`, `response.audio.delta` / `done`, `response.done` | `response.output_item.created`, `response.output_audio.delta` / `done`, `response.completed` | Downstream clients expecting the latest event names need a translation layer. |
| Safety / moderation hooks | Not implemented (`response.warning`, refusal deltas, etc.) | Implemented in official API | Server never surfaces refusal or moderation metadata. |
| Parallel responses | One active `response_id` at a time | Multiple concurrent `response.create` per session | Requests beyond the first are ignored until the in-flight response completes. |

## Additional considerations

* **Input audio**: Clients must continue to send base64 PCM16 @ 24 kHz frames via `input_audio_buffer.append`. No support exists for the newer `input_audio_buffer.commit` payload metadata (`format`, `channels`, etc.).
* **Session configuration**: Fields recently added to `session.update` (tool choice policies, modalities per response, etc.) are not surfaced or stored.
* **Error surface**: The server emits a plain `error` event when upstream services fail. Official spec distinguishes between transport (`response.error`) and moderation-specific warnings.

For an exact reference, compare the generated events in `external-test/realtime/schema.json` (this project) with the schema published by OpenAI in 2024-09 and later.
