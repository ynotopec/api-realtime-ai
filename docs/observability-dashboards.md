# Observability dashboards and alerting

The realtime bridge exports Prometheus metrics for requests, tokens, model calls, and feedback events. This guide explains how to import the curated Grafana dashboard, wire alerting for latency/error/feedback regressions, and tune thresholds per environment.

## Dashboard contents
The provisioning file [`docs/grafana/realtime-observability.json`](./grafana/realtime-observability.json) ships with the following panels:

- **Request latency (p50/p90/p99):** `histogram_quantile()` over `api_request_latency_seconds_bucket`, filterable by `job` and `path`.
- **Token throughput (tokens/sec):** `sum by (direction, model) (rate(api_token_usage_total[1m]))` to catch spikes or drops in prompt/completion tokens.
- **Model call durations:** p95 and mean latency for `model_inference_duration_seconds_bucket` grouped by model.
- **Rolling positive feedback rate:** `positive / total` for `feedback_submissions_total`, grouped by `channel` to separate cohorts.
- **Average feedback rating:** derived from `feedback_score_sum / feedback_score_count` to show mean numeric score when provided.
- **Top errors by route/status:** `topk` over non-2xx `api_requests_total`.
- **Slowest endpoints:** `topk` mean latency per `path` using histogram sums/counts.

Template variables allow selecting Prometheus datasource, `job`, request `path`, and feedback `channel`.

## Importing via Grafana UI
1. Open **Dashboards → Import** in Grafana.
2. Upload `docs/grafana/realtime-observability.json` or paste its contents.
3. Pick the Prometheus datasource that scrapes the bridge (or OTEL collector) and save.
4. Override the default time range (dashboard defaults to last 6h) if needed.

## Provisioning as code
For environments that prefer immutable dashboards:

- **Grafana provisioning folder:** Copy `docs/grafana/realtime-observability.json` to `/etc/grafana/provisioning/dashboards/realtime-api.json` and add a dashboard provider yaml pointing at that directory.
- **Helm with sidecar:** If you deploy Grafana via Helm, create a ConfigMap from the dashboard and enable the dashboards sidecar:
  ```bash
  kubectl create configmap realtime-observability-dashboard \
    --from-file=realtime-observability.json=docs/grafana/realtime-observability.json \
    -n monitoring
  ```
  Add annotations `grafana_dashboard: "1"` (or the label your sidecar expects) so the sidecar mounts it automatically.
- **CI/CD:** Commit the dashboard file and reference it in your manifests to keep Grafana in sync with version control.

## Alerting rules
Prometheus alerting rules are defined in [`docs/prometheus-alerts.yaml`](./prometheus-alerts.yaml):

- **ApiHighLatencyP99:** p99 `api_request_latency_seconds` above 1s for 10m.
- **ApiErrorRateHigh:** non-2xx error ratio above 5% for 10m.
- **TokenThroughputAnomaly:** tokens/sec drop below 50% of the 1h baseline for 15m.
- **ModelCallDurationHigh:** p95 upstream model latency above 5s.
- **PositiveFeedbackRateDrop:** positive feedback rate below 60% for 15m.

Load the rules by mounting the YAML into Prometheus (e.g., `rule_files` entry or `PrometheusRule` CRD in Kubernetes). Grafana Alerting can also import the same expressions—paste them into Grafana’s unified alerting editor and attach contact points.

## Tuning thresholds per environment
- **Dev:** Raise noise tolerance (e.g., latency threshold 2–3s, error rate 20%) while feature flags churn. Consider disabling paging severity and routing alerts to a Slack channel.
- **Staging:** Use thresholds close to prod but shorter `for:` durations to catch regressions before release. Keep paging off but open tickets automatically.
- **Prod:** Tighten thresholds (1s p99, 5% errors, 60% positive rate) and page on sustained breaches. Token throughput anomalies are useful for traffic drops or model outages.

If you deploy via Helm/Kubernetes, parameterize the rule values using environment-specific `values-*.yaml` and render the alert YAML through templating, or maintain separate ConfigMaps per cluster. For Grafana-based alerts, create folders per environment and override thresholds through the UI variables before saving the rule.

## Metric references
Key Prometheus series used by the dashboard and alerts:

- `api_requests_total`, `api_request_latency_seconds_bucket` (HTTP traffic + latency)
- `api_token_usage_total` (prompt/completion token throughput)
- `model_inference_duration_seconds_bucket` (upstream model latency)
- `feedback_submissions_total`, `feedback_submission_latency_seconds_bucket`, `feedback_score_sum`, `feedback_score_count` (feedback quality + latency)

Make sure your Prometheus scrape config includes the `/metrics` endpoint exposed by the FastAPI app or the OTEL collector that forwards these metrics.
