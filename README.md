# Realtime Speech Bridge

Bridge temps réel **audio ↔ transcription ↔ réponse LLM ↔ synthèse vocale** via une API WebSocket compatible OpenAI (`/v1/realtime`) et une API REST (`/v1/chat/completions`).

## Démarrage en moins de 10 minutes

### 1) Préparer l'environnement
```bash
cp .env.example .env
# renseigner OPENAI_API_KEY dans .env
```

### 2) Installation déterministe (même procédure pour tous)
```bash
make install
```

### 3) Lancer en une commande
```bash
make run
```

Serveur démarré par défaut sur `http://0.0.0.0:8080`.

## Exemple d'entrée / sortie reproductible

Une fois le service lancé, tester un endpoint local qui ne dépend pas d'un flux audio :

```bash
curl -s http://127.0.0.1:8080/api/transcripts | jq
```

Sortie attendue (exemple initial) :
```json
[]
```

## Artefacts de documentation

- Vue d'ensemble technique : `docs/overview.md`
- Architecture : `docs/architecture.md`
- Cas d'usage métier : `USE_CASE.md`
- Valeur mesurable : `VALUE.md`
- Statut d'innovation : `INNOVATION_STATUS.md`

## Fonctionnalités principales

- WebSocket realtime: `/v1/realtime`
- Chat completions compatible OpenAI: `/v1/chat/completions`
- Persistance des transcripts SQLite + replay: `/api/transcripts/*`
- Observabilité: logs structurés, traces OTEL, métriques Prometheus `/metrics`

## Déploiement et opérations

- Helm chart: `charts/realtime-bridge/`
- Documentation infra/ops: `docs/deployment.md`, `docs/load-testing.md`, `docs/observability-dashboards.md`
