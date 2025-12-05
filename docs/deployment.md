# Deployment Guide

This repository now ships Dockerfiles for CPU and GPU builds, Helm charts for Kubernetes, and CI/CD automation for promoting changes from development to production clusters.

## Container builds

- **CPU image:** `Dockerfile` (Python 3.11 slim, venv at `/opt/venv`, includes `ffmpeg`).
- **GPU image:** `Dockerfile.gpu` (CUDA 12.3 runtime, Python 3.11 venv, `ffmpeg`).

Both images run `uvicorn app:app --ws websockets` and expect environment variables documented in `README.md`.

### Local build examples

```bash
# CPU build
REGISTRY=ghcr.io/your-org/api-realtime-ai
TAG=dev

docker build -f Dockerfile -t "$REGISTRY:$TAG" .

# GPU build
GPU_TAG=dev-gpu
docker build -f Dockerfile.gpu -t "$REGISTRY:$GPU_TAG" .
```

## Kubernetes (Helm)

The chart lives in `charts/realtime-bridge/`. Use the provided values files for dev/prod:

```bash
helm upgrade --install realtime-bridge ./charts/realtime-bridge \
  -f charts/realtime-bridge/values-dev.yaml \
  --set image.repository=$REGISTRY --set image.tag=$TAG \
  --set secrets.openaiApiKey="$OPENAI_API_KEY" \
  --set secrets.apiTokens="$API_TOKENS"
```

- **GPU scheduling:** Set `gpu.enabled=true` (enabled in `values-prod.yaml`) to switch to the CUDA image and apply GPU node selectors/tolerations/resource limits.
- **Ingress:** Enabled by default; configure host/TLS in the values files.
- **Autoscaling:** HPA is enabled with CPU and optional memory targets.
- **Config/Secrets:** API endpoints and VAD defaults come from ConfigMaps, while API keys and tokens come from a Kubernetes Secret.

## CI/CD

`.github/workflows/deploy.yml` builds and pushes both CPU and GPU images to GHCR, deploys to the dev cluster via Helm, waits for rollout, and then promotes to production after verification/approval.

Required secrets:

- `GHCR_USERNAME` / `GHCR_TOKEN` for registry login.
- `KUBE_CONFIG_DEV` and `KUBE_CONFIG_PROD` (base64-encoded kubeconfigs).
- `OPENAI_API_KEY`, `STT_API_KEY`, `TTS_API_KEY`, `API_TOKENS` for templating Helm secrets.

The prod deployment job is attached to the `production` environment for manual approval before applying manifests.
