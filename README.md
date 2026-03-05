# ⚡ LLM Streaming Skill-Up 2025

Hands-on implementation of **Streaming Responses & Low-Latency LLM Serving**  
powered by the **Groq API** — one POC + Blog + Implementation per quarter.

---

## 📁 Project Structure

```
llm_streaming/
├── .env                          ← API keys & config (edit this first!)
├── requirements.txt
├── run.py                        ← Unified CLI entry point
│
├── shared/
│   ├── config.py                 ← Loads .env, exposes cfg object
│   ├── groq_client.py            ← Async/sync Groq client factory
│   └── metrics.py                ← Prometheus counters / histograms
│
├── q1_foundations/
│   ├── streaming.py              ← Core SSE streaming engine
│   └── server.py                 ← FastAPI server + embedded chat UI
│
├── q2_benchmarking/
│   └── benchmark.py              ← Throughput & latency benchmark suite
│
├── q3_structured/
│   ├── structured_stream.py      ← JSON schema-constrained streaming
│   └── router.py                 ← Intelligent multi-model router
│
└── q4_observability/
    ├── production_server.py      ← Full production FastAPI server
    └── locustfile.py             ← Locust load testing
```

---

## 🚀 Quick Start

### 1. Set up your environment

```bash
# Clone / download this project
cd llm_streaming

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key to .env
# Get one free at: https://console.groq.com
nano .env   # Set GROQ_API_KEY=gsk_...
```

### 2. Run a quick demo (no server needed)

```bash
python run.py demo
```

---

## 📋 Q1 — Foundations: Streaming Chat Server

**What it does:** FastAPI server with SSE streaming, TTFT measurement, and a full chat UI.

```bash
python run.py q1
# Open: http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Key files:**
- `q1_foundations/streaming.py` — Core streaming engine with metrics
- `q1_foundations/server.py` — FastAPI server + built-in chat UI

**API call example:**
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}' \
  --no-buffer
```

**SSE events you'll receive:**
```
data: {"event": "ttft",  "ms": 245.3}
data: {"event": "token", "text": "Hello"}
data: {"event": "token", "text": " there!"}
data: {"event": "done",  "tokens": 12, "tps": 63.2, "total_ms": 1380}
```

---

## 🔬 Q2 — Benchmarking: Throughput & Latency

**What it does:** Runs a benchmark matrix across models and concurrency levels.

```bash
python run.py q2
```

Produces a rich terminal table + saves `q2_benchmarking/benchmark_report.json`:

```
╭──────────────────────────┬──────────┬───────────┬──────────────┬──────────────╮
│ Model                    │ Requests │ Success % │ TTFT P50(ms) │ Mean Tok/s   │
├──────────────────────────┼──────────┼───────────┼──────────────┼──────────────┤
│ llama3-8b-8192           │    10    │  100.0%   │     210      │    72.4      │
│ llama3-8b-8192 (c=5)     │    20    │  100.0%   │     230      │   198.1      │
│ llama-3.3-70b-versatile  │    10    │  100.0%   │     340      │    45.2      │
╰──────────────────────────┴──────────┴───────────┴──────────────┴──────────────╯
```

---

## 🧩 Q3 — Structured Output Streaming

**What it does:** Streams JSON-schema-constrained output token by token.

```bash
# Demo: entity extraction, sentiment analysis, code review
python run.py q3

# Demo: intelligent model router
python run.py q3-router
```

**Using the structured extractor in your code:**
```python
from q3_structured.structured_stream import extract_entities, analyze_sentiment

# Entity extraction
entities = await extract_entities("Apple CEO Tim Cook announced in Cupertino...")
# → {"people": ["Tim Cook"], "organizations": ["Apple"], "locations": ["Cupertino"], ...}

# Sentiment analysis
sentiment = await analyze_sentiment("This product is absolutely fantastic!")
# → {"sentiment": "positive", "score": 0.92, "confidence": 0.95, ...}
```

**Router automatically selects the right model:**
```python
from q3_structured.router import LLMRouter

router = LLMRouter()
async for event in router.stream(messages, requires_json=True):
    print(event)
# → {"event": "routing", "model": "llama3-70b-8192", "reason": "structured JSON required"}
# → {"event": "ttft",    "ms": 180.4}
# → {"event": "token",   "text": "{"}
# ...
```

---

## ⚙️ Q4 — Production Server with Observability

**What it does:** Production FastAPI server with Prometheus metrics, circuit breaker, prefix cache, and structured logging.

```bash
python run.py q4
```

**Available endpoints:**

| Endpoint | Description |
|---|---|
| `POST /v1/chat/stream` | SSE streaming (full feature set) |
| `POST /v1/chat/completions` | OpenAI-compatible alias |
| `GET  /health` | Health check + circuit status |
| `GET  /metrics` | Prometheus metrics |
| `GET  /stats` | Human-readable stats |
| `GET  /docs` | Swagger UI |

**Run load test with Locust:**
```bash
# Start server first: python run.py q4

# Then in another terminal:
locust -f q4_observability/locustfile.py \
       --host http://localhost:8000 \
       -u 20 -r 2 --run-time 2m --headless
```

**Key metrics exposed (Prometheus):**
```
llm_ttft_seconds{model="..."}           # Time to first token histogram
llm_e2e_seconds{model="..."}            # End-to-end latency histogram
llm_tokens_generated_total{model="..."}  # Token throughput counter
llm_active_streams                       # Concurrent connections gauge
llm_prefix_cache_hits_total             # Cache efficiency counter
```

---

## 🔧 Configuration (.env)

```ini
GROQ_API_KEY=gsk_...               # Required: your Groq API key

GROQ_MODEL_FAST=llama3-8b-8192             # Fast model (Q1, Q2)
GROQ_MODEL_CAPABLE=llama-3.3-70b-versatile # Capable model (Q2, Q3)
GROQ_MODEL_STRUCTURED=llama3-70b-8192      # Structured output (Q3)

APP_PORT=8000
STREAM_TIMEOUT_SECONDS=60
MAX_TOKENS_DEFAULT=1024
ENABLE_METRICS=true
```

**Supported Groq models:**
| Model ID | Context | Best for |
|---|---|---|
| `llama3-8b-8192` | 8K | Fast, low latency |
| `llama-3.3-70b-versatile` | 128K | Complex reasoning |
| `llama3-70b-8192` | 8K | Structured output |
| `mixtral-8x7b-32768` | 32K | Long context |
| `gemma2-9b-it` | 8K | Instruction following |

---

## 🏃 Run All Quarters (in order)

```bash
# Terminal 1: Run the Q1 server
python run.py q1

# Terminal 2: Test with curl
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is vLLM?"}]}' \
  --no-buffer

# Run benchmarks (no server needed)
python run.py q2

# Run structured output demo (no server needed)
python run.py q3

# Run router demo (no server needed)
python run.py q3-router

# Run production server
python run.py q4
```
