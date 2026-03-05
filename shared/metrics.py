"""
shared/metrics.py
─────────────────
Prometheus metrics registry + helper functions shared across all quarters.
Exposes an HTTP /metrics endpoint on a background thread.
"""
import time
import threading
from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server, REGISTRY
)
from shared.config import cfg


# ── Counters ──────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["model", "status"],           # labels
)

TOKEN_COUNT = Counter(
    "llm_tokens_total",
    "Total tokens generated",
    ["model", "type"],             # type: input | output
)

# ── Histograms ────────────────────────────────────────────────────────────────
TTFT_HISTOGRAM = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token (seconds)",
    ["model"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)

E2E_LATENCY = Histogram(
    "llm_e2e_latency_seconds",
    "End-to-end request latency (seconds)",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)

TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second",
    "Token generation throughput",
    ["model"],
    buckets=[10, 20, 50, 100, 150, 200, 300, 500],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of currently active streaming requests",
)

QUEUE_DEPTH = Gauge(
    "llm_queue_depth",
    "Simulated queue depth (waiting requests)",
)


# ── Helper context manager ────────────────────────────────────────────────────
class RequestTracker:
    """
    Use as a context manager to auto-track request lifecycle.

    Usage:
        async with RequestTracker(model="llama3-8b-8192") as t:
            ... do streaming ...
            t.record_first_token()
    """

    def __init__(self, model: str):
        self.model = model
        self._start: float = 0.0
        self._first_token_time: float | None = None
        self._token_count: int = 0

    def __enter__(self):
        self._start = time.perf_counter()
        ACTIVE_REQUESTS.inc()
        return self

    def __exit__(self, exc_type, *_):
        elapsed = time.perf_counter() - self._start
        status = "error" if exc_type else "success"
        REQUEST_COUNT.labels(model=self.model, status=status).inc()
        E2E_LATENCY.labels(model=self.model).observe(elapsed)
        if self._token_count > 0 and elapsed > 0:
            TOKENS_PER_SECOND.labels(model=self.model).observe(
                self._token_count / elapsed
            )
        ACTIVE_REQUESTS.dec()

    def record_first_token(self) -> float:
        """Call when the first token arrives. Returns TTFT in ms."""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            ttft = self._first_token_time - self._start
            TTFT_HISTOGRAM.labels(model=self.model).observe(ttft)
            return ttft * 1000  # ms
        return 0.0

    def record_token(self, count: int = 1):
        self._token_count += count
        TOKEN_COUNT.labels(model=self.model, type="output").inc(count)


def start_metrics_server():
    """Start Prometheus metrics HTTP server on a background thread."""
    if not cfg.ENABLE_METRICS:
        return
    port = cfg.PROMETHEUS_PORT
    thread = threading.Thread(
        target=lambda: start_http_server(port),
        daemon=True,
    )
    thread.start()
    print(f"📊  Prometheus metrics → http://localhost:{port}/metrics")