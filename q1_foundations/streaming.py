"""
q1_foundations/streaming.py
────────────────────────────
Q1 — Core streaming engine.
Handles SSE generation, TTFT measurement, and error recovery.
"""
import sys, json, asyncio, time, logging
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIError, APITimeoutError
from shared.config import cfg
from shared.groq_client import get_async_client
from shared.metrics import RequestTracker

logger = logging.getLogger(__name__)


# ─── SSE helper ───────────────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(payload)}\n\n"


# ─── Core streaming function ──────────────────────────────────────────────────
async def stream_chat(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """
    Yields SSE-formatted strings:
      {"event": "ttft",   "ms": 245.3}
      {"event": "token",  "text": "Hello"}
      {"event": "done",   "tokens": 87, "tps": 63.2, "total_ms": 1380}
      {"event": "error",  "msg": "..."}

    The client reads these with an EventSource or fetch+ReadableStream.
    """
    _model      = model       or cfg.MODEL_FAST
    _max_tokens = max_tokens  or cfg.MAX_TOKENS
    _temp       = temperature or cfg.TEMPERATURE

    # Prepend system prompt if provided
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    client = get_async_client()

    with RequestTracker(model=_model) as tracker:
        try:
            async with asyncio.timeout(cfg.STREAM_TIMEOUT):
                stream = await client.chat.completions.create(
                    model=_model,
                    messages=full_messages,
                    max_tokens=_max_tokens,
                    temperature=_temp,
                    stream=True,
                )

                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue

                    # First token → emit TTFT metric
                    if tracker._first_token_time is None:
                        ttft_ms = tracker.record_first_token()
                        yield _sse({"event": "ttft", "ms": round(ttft_ms, 2)})

                    tracker.record_token()
                    yield _sse({"event": "token", "text": delta})

                # Final summary event
                elapsed_ms = (time.perf_counter() - tracker._start) * 1000
                tps = tracker._token_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                yield _sse({
                    "event":    "done",
                    "tokens":   tracker._token_count,
                    "tps":      round(tps, 1),
                    "total_ms": round(elapsed_ms, 1),
                })

        except asyncio.TimeoutError:
            logger.error("Stream timed out after %ss", cfg.STREAM_TIMEOUT)
            yield _sse({"event": "error", "msg": f"Request timed out after {cfg.STREAM_TIMEOUT}s"})

        except APITimeoutError as e:
            logger.error("Groq API timeout: %s", e)
            yield _sse({"event": "error", "msg": "Groq API timed out"})

        except APIError as e:
            logger.error("Groq API error: %s", e)
            yield _sse({"event": "error", "msg": str(e)})

        except Exception as e:
            logger.exception("Unexpected streaming error")
            yield _sse({"event": "error", "msg": "Internal server error"})