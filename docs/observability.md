# Observability, dashboards, and alerting

This service exports Prometheus metrics for HTTP requests, token usage, upstream model calls, and user feedback quality. Use the provided Grafana dashboard JSON and Prometheus alert rules to keep latency, reliability, throughput, and customer sentiment in guardrails.

## Metrics of interest

* **Request latency histogram**: `api_request_latency_seconds_bucket` (percentiles p50/p90/p99).
* **Request volume/error counters**: `api_requests_total{status}`.
* **Token throughput**: `api_token_usage_total{direction,model}`.
* **Model latency histogram**: `model_inference_duration_seconds_bucket`.
* **Feedback signals**:
  * `feedback_events_total{channel,outcome}` for positive/negative counts.
  * `feedback_score_sum{channel}` and `feedback_score_count{channel}` to derive average rating/score.

## Grafana dashboard

A ready-to-import dashboard lives at [`docs/grafana/realtime-observability-dashboard.json`](./grafana/realtime-observability-dashboard.json). Panels include:

* Request latency percentiles (p50/p90/p99) from Prometheus histograms.
* Token throughput (tokens/sec) by direction/model and upstream model call durations (p50/p95/p99).
* Feedback quality score panels: rolling positive rate and average rating/score per channel.
* Top errors by route/status and a table of slowest endpoints by p99 latency.

### Provisioning steps

1. Copy the JSON file into your Grafana provisioning directory, e.g. `provisioning/dashboards/realtime-observability.json`.
2. Add a provisioning entry (if not present):
   ```yaml
   apiVersion: 1
   providers:
     - name: realtime-api
       orgId: 1
       folder: Realtime API
       type: file
       disableDeletion: false
       editable: true
       options:
         path: /etc/grafana/provisioning/dashboards
   ```
3. Restart/reload Grafana. The dashboard will bind to the `prometheus` datasource UID; adjust the UID in the JSON if your datasource name differs.
4. Use the `channel` templating variable to filter feedback quality panels to a specific customer/channel.

## Prometheus/Grafana alerting

Alert rules are templated in Helm at [`charts/realtime-bridge/templates/alerts.yaml`](../charts/realtime-bridge/templates/alerts.yaml) and keyed off the metrics above:

* **High latency**: p99 `api_request_latency_seconds` above threshold for a sustained window.
* **Elevated error rate**: 5xx ratio above threshold.
* **Token throughput anomalies**: tokens/sec drops below a baseline.
* **Feedback sentiment drop**: rolling positive rate falls below baseline.

These rules assume the Prometheus Operator (`PrometheusRule` CRD). Grafana-managed alerts can point at the same PromQL expressions if you prefer Grafana Alerting instead of Prometheus.

### Tuning thresholds per environment

Thresholds and windows are overridable via Helm values. Defaults live in [`values.yaml`](../charts/realtime-bridge/values.yaml); dev/prod overrides are in [`values-dev.yaml`](../charts/realtime-bridge/values-dev.yaml) and [`values-prod.yaml`](../charts/realtime-bridge/values-prod.yaml).

Examples:

* **Dev**: p99 latency 2.5s, error ratio 10%, positive feedback floor 60% to reduce noise during iteration.
* **Prod**: p99 latency 1s, error ratio 3%, positive feedback floor 80%, stricter throughput floor.

Update the relevant `monitoring.alerts.*` values before deploying:
```bash
helm upgrade --install realtime charts/realtime-bridge \
  -f charts/realtime-bridge/values-prod.yaml \
  --set monitoring.alerts.latency.p99Seconds=0.8 \
  --set monitoring.alerts.feedbackPositiveRate.minRate=0.85
```

## Operational tips

* If you run Grafana inside the chart, mount `docs/grafana/realtime-observability-dashboard.json` into the dashboards directory via a `ConfigMap` or your Grafana subchart.
* When using Grafana Alerting, reuse the PromQL from the PrometheusRule file and set the same `for` durations to avoid alert flaps.
* The feedback metrics treat boolean thumbs-up/down as scores (1/0) but also accept a numeric `rating` in the feedback payload; the average rating panel reflects whichever data is available.
