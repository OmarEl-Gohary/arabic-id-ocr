"""Prometheus metrics for the OCR serving API."""

from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest

REQUEST_COUNT = Counter(
    "ocr_requests_total",
    "Total number of OCR requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ocr_request_duration_seconds",
    "OCR request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

DETECTION_COUNT = Histogram(
    "ocr_fields_detected_per_request",
    "Number of fields detected per request",
    buckets=[0, 1, 3, 5, 8, 10, 15],
)

OCR_CONFIDENCE = Histogram(
    "ocr_field_confidence",
    "OCR confidence scores per field",
    ["field_name"],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

EMPTY_FIELDS = Counter(
    "ocr_empty_fields_total",
    "Number of fields with no detected text",
    ["field_name"],
)

MODEL_LOADED = Gauge(
    "ocr_model_loaded",
    "Whether the OCR model is currently loaded (1=yes, 0=no)",
)


def record_request(endpoint: str, status: str, duration: float):
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_detection(num_fields: int):
    DETECTION_COUNT.observe(num_fields)


def record_ocr_result(field_name: str, confidence: float, text: str):
    if confidence > 0:
        OCR_CONFIDENCE.labels(field_name=field_name).observe(confidence)
    if not text:
        EMPTY_FIELDS.labels(field_name=field_name).inc()


def get_metrics() -> bytes:
    return generate_latest()
