# Architecture

```mermaid
flowchart LR
Client[Client Web / Mobile] -->|PCM16 24kHz| WS[/v1/realtime WS/]
WS --> VAD[Server VAD]
VAD --> STT[STT API]
STT --> LLM[LLM Chat API]
LLM --> TTS[TTS API]
TTS --> WS
WS -->|audio delta + text delta| Client

WS --> DB[(SQLite transcripts.db)]
DB --> Replay[/api/transcripts/*/]
```

## Décisions clés

- FastAPI pour unifier API REST et WebSocket.
- Pipeline externalisé (STT/LLM/TTS) pour flexibilité fournisseur.
- SQLite local pour traçabilité/rejeu sans infra lourde.
- Métriques/OTEL natifs pour observabilité de bout en bout.
