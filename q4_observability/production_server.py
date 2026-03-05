"""
q4_observability/production_server.py
───────────────────────────────────────
Q4 — Production-grade FastAPI server with:
  • Prometheus metrics (TTFT, throughput, queue depth, errors)
  • Request ID tracing
  • Prefix cache simulation (tracks repeated system prompts)
  • Circuit breaker pattern
  • Structured logging

Run:
    cd llm_streaming
    uvicorn q4_observability.production_server:app --port 8000 --workers 1
"""
import sys, json, asyncio, time, hashlib, logging
from pathlib import Path
from collections import defaultdict, deque
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge

from shared.config import cfg
from shared.groq_client import get_async_client
from q3_structured.router import LLMRouter, classify_request

# ─── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()


# ─── Prometheus metrics ───────────────────────────────────────────────────────
REQ_COUNT = Counter(
    "api_requests_total", "Total API requests",
    ["method", "endpoint", "status_code"]
)
TTFT_HIST = Histogram(
    "llm_ttft_seconds", "Time to first token",
    ["model"],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0]
)
E2E_HIST = Histogram(
    "llm_e2e_seconds", "End-to-end latency",
    ["model"],
    buckets=[0.5, 1, 2, 3, 5, 10, 20, 30]
)
TOKENS_COUNTER = Counter(
    "llm_tokens_generated_total", "Tokens generated",
    ["model"]
)
ACTIVE_STREAMS = Gauge("llm_active_streams", "Active streaming connections")
QUEUE_DEPTH    = Gauge("llm_queue_depth",    "Simulated request queue depth")
CACHE_HITS     = Counter("llm_prefix_cache_hits_total",   "Prefix cache hits")
CACHE_MISSES   = Counter("llm_prefix_cache_misses_total", "Prefix cache misses")


# ─── Prefix Cache (simulated) ─────────────────────────────────────────────────
class PrefixCache:
    """
    Tracks system prompt hashes.
    In production vLLM, this corresponds to KV cache reuse.
    Here we simulate the TTFT speedup: cached prefixes get ~70% lower TTFT.
    """
    def __init__(self, max_entries: int = 100):
        self._cache: dict[str, float] = {}   # hash → first-seen timestamp
        self._max = max_entries

    def lookup(self, system_prompt: str | None) -> bool:
        if not system_prompt:
            return False
        key = hashlib.md5(system_prompt.encode()).hexdigest()
        if key in self._cache:
            CACHE_HITS.inc()
            return True
        self._cache[key] = time.time()
        if len(self._cache) > self._max:
            oldest = min(self._cache, key=self._cache.get)
            del self._cache[oldest]
        CACHE_MISSES.inc()
        return False

    @property
    def size(self) -> int:
        return len(self._cache)


# ─── Circuit Breaker ──────────────────────────────────────────────────────────
class CircuitBreaker:
    """
    Simple circuit breaker: after `threshold` failures in `window` seconds,
    open the circuit and reject requests for `recovery` seconds.
    """
    def __init__(self, threshold: int = 5, window: float = 60.0, recovery: float = 30.0):
        self._failures: deque = deque()
        self.threshold = threshold
        self.window    = window
        self.recovery  = recovery
        self._open_since: float | None = None

    @property
    def is_open(self) -> bool:
        if self._open_since is None:
            return False
        if time.time() - self._open_since > self.recovery:
            self._open_since = None   # Auto-close after recovery period
            self._failures.clear()
            log.info("circuit_breaker_closed")
            return False
        return True

    def record_failure(self):
        now = time.time()
        self._failures.append(now)
        # Evict old failures outside window
        while self._failures and now - self._failures[0] > self.window:
            self._failures.popleft()
        if len(self._failures) >= self.threshold:
            self._open_since = now
            log.warning("circuit_breaker_opened", failures=len(self._failures))

    def record_success(self):
        # Gradually clear failures on success
        if self._failures:
            self._failures.popleft()


# ─── App state ────────────────────────────────────────────────────────────────
prefix_cache   = PrefixCache()
circuit        = CircuitBreaker(threshold=5, window=60)
router         = LLMRouter()
_request_log   = deque(maxlen=500)      # In-memory rolling request log


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.validate()
    log.info("server_starting", host=cfg.HOST, port=cfg.PORT, env=cfg.ENV)
    yield
    log.info("server_shutdown")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Production Server — Q4",
    description="Production-grade streaming API with full observability",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: request logging + metrics ─────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000

    REQ_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    log.info(
        "http_request",
        method=request.method,
        path=str(request.url.path),
        status=response.status_code,
        ms=round(elapsed, 1),
    )
    return response


# ─── Pydantic models ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    system_prompt: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    requires_json: bool = False
    request_id: str | None = None   # Optional client-provided ID


# ─── Streaming endpoint ───────────────────────────────────────────────────────
@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    # Circuit breaker check
    if circuit.is_open:
        raise HTTPException(503, "Service temporarily unavailable (circuit open)")

    request_id = req.request_id or f"req_{int(time.time() * 1000)}"
    messages = [m.model_dump() for m in req.messages]

    # Prepend system prompt
    if req.system_prompt:
        messages = [{"role": "system", "content": req.system_prompt}] + messages

    # Check prefix cache
    cache_hit = prefix_cache.lookup(req.system_prompt)

    # Route decision (for logging)
    decision = classify_request(messages, req.requires_json)

    log.info(
        "stream_start",
        request_id=request_id,
        model=decision.model,
        route=decision.route.value,
        cache_hit=cache_hit,
    )

    async def event_stream():
        ACTIVE_STREAMS.inc()
        QUEUE_DEPTH.inc()

        start = time.perf_counter()
        ttft_ms: float | None = None
        token_count = 0

        try:
            QUEUE_DEPTH.dec()

            # Yield routing info to client
            yield f"data: {json.dumps({'event': 'routing', 'model': decision.model, 'route': decision.route.value, 'cache_hit': cache_hit, 'request_id': request_id})}\n\n"

            async for event in router.stream(
                messages,
                requires_json=req.requires_json,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ):
                if event["event"] == "routing":
                    continue   # Already sent above
                elif event["event"] == "ttft":
                    ttft_ms = event["ms"] / 1000
                    # Apply cache speedup simulation
                    simulated_ttft = ttft_ms * (0.3 if cache_hit else 1.0)
                    TTFT_HIST.labels(model=decision.model).observe(simulated_ttft)
                    yield f"data: {json.dumps({**event, 'cache_hit': cache_hit})}\n\n"
                elif event["event"] == "token":
                    token_count += 1
                    TOKENS_COUNTER.labels(model=decision.model).inc()
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["event"] == "done":
                    elapsed = time.perf_counter() - start
                    E2E_HIST.labels(model=decision.model).observe(elapsed)
                    circuit.record_success()
                    log.info(
                        "stream_done",
                        request_id=request_id,
                        tokens=token_count,
                        total_ms=round(elapsed * 1000, 1),
                        cache_hit=cache_hit,
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["event"] == "error":
                    circuit.record_failure()
                    log.error("stream_error", request_id=request_id, msg=event["msg"])
                    yield f"data: {json.dumps(event)}\n\n"

        except asyncio.CancelledError:
            log.info("stream_cancelled", request_id=request_id)
        finally:
            ACTIVE_STREAMS.dec()
            if QUEUE_DEPTH._value.get() > 0:
                QUEUE_DEPTH.dec()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":      "no-cache",
            "X-Accel-Buffering":  "no",
            "X-Request-ID":       request_id,
        },
    )


# ─── Observability endpoints ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":         "ok" if not circuit.is_open else "degraded",
        "circuit":        "open" if circuit.is_open else "closed",
        "cache_size":     prefix_cache.size,
        "active_streams": ACTIVE_STREAMS._value.get(),
        "env":            cfg.ENV,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/stats")
async def stats():
    """Human-readable stats dashboard."""
    return {
        "routing_stats":  router.routing_stats(),
        "prefix_cache":   {"size": prefix_cache.size},
        "circuit":        {"open": circuit.is_open},
        "active_streams": ACTIVE_STREAMS._value.get(),
    }


# ─── OpenAI-compatible alias ──────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def openai_compat(req: Request):
    """
    OpenAI-compatible endpoint — drop-in for existing apps.
    Internally routes through our production stack.
    """
    body = await req.json()
    messages = [Message(**m) for m in body.get("messages", [])]
    stream_req = ChatRequest(
        messages=messages,
        model=body.get("model"),
        max_tokens=body.get("max_tokens", 1024),
        temperature=body.get("temperature", 0.7),
    )
    return await chat_stream(stream_req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "q4_observability.production_server:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=(cfg.ENV == "development"),
        log_level="info",
    )