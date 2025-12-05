# Realtime Bridge Helm Chart

This chart deploys the Realtime Speech Bridge FastAPI service. It supports CPU- and GPU-backed workloads, ingress, autoscaling, and configuration through ConfigMaps and Secrets.

## Usage

```bash
helm upgrade --install realtime-bridge ./charts/realtime-bridge \
  -f charts/realtime-bridge/values-dev.yaml \  # or values-prod.yaml
  --set image.repository=ghcr.io/your-org/api-realtime-ai \
  --set image.tag=dev
```

Override secret values at install time to avoid committing credentials:

```bash
helm upgrade --install realtime-bridge ./charts/realtime-bridge \
  -f charts/realtime-bridge/values-prod.yaml \
  --set secrets.openaiApiKey="$OPENAI_API_KEY" \
  --set secrets.sttApiKey="$STT_API_KEY" \
  --set secrets.ttsApiKey="$TTS_API_KEY" \
  --set secrets.apiTokens="$API_TOKENS"
```

GPU scheduling is enabled by setting `gpu.enabled=true`, which switches the deployment to the CUDA-based image and adds node selectors/tolerations defined in the values file.
