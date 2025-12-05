import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from pythonjsonlogger import jsonlogger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_tracing_configured = False
_metrics_configured = False
_logging_configured = False

REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TOKEN_COUNTER = Counter(
    "api_token_usage_total",
    "Number of tokens processed",
    ["direction", "model"],
)
FEEDBACK_COUNTER = Counter(
    "feedback_events_total",
    "Feedback submissions by outcome and channel",
    ["channel", "outcome"],
)
FEEDBACK_SCORE_SUM = Counter(
    "feedback_score_sum",
    "Cumulative feedback score for calculating averages",
    ["channel"],
)
FEEDBACK_SCORE_COUNT = Counter(
    "feedback_score_count",
    "Number of feedback scores recorded",
    ["channel"],
)
MODEL_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Duration of upstream model inference calls",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
INFLIGHT_REQUESTS = Gauge(
    "api_inflight_requests",
    "Current number of in-flight API requests",
)


def configure_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(trace_id)s %(span_id)s"
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    class RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.request_id = _request_id_ctx.get() or "-"
            span_ctx = trace.get_current_span().get_span_context()
            if span_ctx and span_ctx.is_valid:
                record.trace_id = format(span_ctx.trace_id, "032x")
                record.span_id = format(span_ctx.span_id, "016x")
            else:
                record.trace_id = "-"
                record.span_id = "-"
            return True

    root_logger.addFilter(RequestIdFilter())
    _logging_configured = True


def configure_tracing(service_name: str = "api-realtime-ai") -> None:
    global _tracing_configured
    if _tracing_configured:
        return
    sampling_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_RATIO", "1.0"))
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource, sampler=TraceIdRatioBased(sampling_ratio))
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    span_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    _tracing_configured = True


def configure_metrics(service_name: str = "api-realtime-ai") -> None:
    global _metrics_configured
    if _metrics_configured:
        return
    resource = Resource.create({"service.name": service_name})
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    metric_exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics")
    reader = PeriodicExportingMetricReader(metric_exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _metrics_configured = True


def inject_tracing_headers(headers: Dict[str, str]) -> None:
    propagate.inject(headers)


def record_token_usage(count: int, direction: str, model: Optional[str]) -> None:
    TOKEN_COUNTER.labels(direction=direction, model=model or "unknown").inc(count)


def record_model_inference(duration_seconds: float, model: Optional[str]) -> None:
    MODEL_DURATION.labels(model=model or "unknown").observe(duration_seconds)


def record_feedback(channel: str, positive: bool, rating: Optional[float] = None) -> None:
    outcome = "positive" if positive else "negative"
    FEEDBACK_COUNTER.labels(channel=channel, outcome=outcome).inc()

    score = rating if rating is not None else (1 if positive else 0)
    FEEDBACK_SCORE_SUM.labels(channel=channel).inc(score)
    FEEDBACK_SCORE_COUNT.labels(channel=channel).inc()


def _build_request_log_fields(
    request: Request,
    response_status: int,
    duration: float,
) -> Dict[str, Any]:
    user_id = request.headers.get("x-user-id") or request.headers.get("x-client-id")
    model_id = request.headers.get("x-model-id")
    token_header = request.headers.get("x-token-count")
    try:
        tokens = int(token_header) if token_header is not None else None
    except ValueError:
        tokens = None
    return {
        "path": request.url.path,
        "method": request.method,
        "status": response_status,
        "latency_ms": round(duration * 1000, 3),
        "user_id": user_id,
        "model_id": model_id,
        "tokens": tokens,
    }


def add_observability(app: FastAPI) -> None:
    configure_logging()
    configure_tracing()
    configure_metrics()

    tracer = trace.get_tracer(__name__)

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):  # type: ignore[no-redef]
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        _request_id_ctx.set(request_id)
        start = time.perf_counter()
        INFLIGHT_REQUESTS.inc()
        with tracer.start_as_current_span(
            "http.request", attributes={"http.method": request.method, "http.target": request.url.path}
        ):
            response: Optional[Response] = None
            try:
                response = await call_next(request)
                return response
            finally:
                duration = time.perf_counter() - start
                status_code = response.status_code if response is not None else 500
                REQUEST_COUNTER.labels(method=request.method, path=request.url.path, status=status_code).inc()
                REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(duration)
                log_fields = _build_request_log_fields(request, status_code, duration)
                logging.getLogger("request").info("request.complete", extra=log_fields)
                INFLIGHT_REQUESTS.dec()
                if response is not None:
                    response.headers["x-request-id"] = request_id

    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:  # type: ignore[no-redef]
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = [
    "add_observability",
    "inject_tracing_headers",
    "record_token_usage",
    "record_model_inference",
    "record_feedback",
]
