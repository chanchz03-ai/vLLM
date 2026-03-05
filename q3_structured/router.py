"""
q3_structured/router.py
────────────────────────
Q3 — Intelligent multi-model router.
Routes requests to the right Groq model based on complexity,
task type, and whether structured output is needed.

Run:
    cd llm_streaming
    python -m q3_structured.router
"""
import sys, json, asyncio, time
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import cfg
from shared.groq_client import get_async_client

# ─── Route categories ──────────────────────────────────────────────────────────
class RouteType(str, Enum):
    FAST       = "fast"        # Simple / factual → Llama 3 8B
    CAPABLE    = "capable"     # Complex / long   → Llama 3.3 70B
    STRUCTURED = "structured"  # JSON output      → Llama 3 70B


@dataclass
class RoutingDecision:
    route:      RouteType
    model:      str
    reason:     str
    confidence: float   # 0–1


# ─── Routing rules ────────────────────────────────────────────────────────────
_COMPLEX_KEYWORDS = {
    "explain", "analyze", "compare", "design", "architecture",
    "implement", "write code", "debug", "evaluate", "pros and cons",
    "essay", "report", "plan", "strategy",
}

_SIMPLE_KEYWORDS = {
    "what is", "who is", "when did", "where is", "define",
    "how many", "list", "name",
}


def classify_request(
    messages: list[dict],
    requires_json: bool = False,
) -> RoutingDecision:
    """
    Rule-based classifier. In production you'd replace the
    heuristics with a tiny classifier model call.
    """
    if requires_json:
        return RoutingDecision(
            route=RouteType.STRUCTURED,
            model=cfg.MODEL_STRUCTURED,
            reason="structured JSON output required",
            confidence=1.0,
        )

    # Combine all message content for analysis
    full_text = " ".join(m.get("content", "") for m in messages).lower()
    word_count = len(full_text.split())

    # Count keyword signals
    complex_signals = sum(1 for kw in _COMPLEX_KEYWORDS if kw in full_text)
    simple_signals  = sum(1 for kw in _SIMPLE_KEYWORDS  if kw in full_text)

    # Long context → capable model
    if word_count > 200:
        return RoutingDecision(
            route=RouteType.CAPABLE,
            model=cfg.MODEL_CAPABLE,
            reason=f"long context ({word_count} words)",
            confidence=0.9,
        )

    # Complex task signals
    if complex_signals > 1 or (complex_signals == 1 and word_count > 50):
        return RoutingDecision(
            route=RouteType.CAPABLE,
            model=cfg.MODEL_CAPABLE,
            reason=f"complex task ({complex_signals} signals: {[kw for kw in _COMPLEX_KEYWORDS if kw in full_text][:3]})",
            confidence=0.8,
        )

    # Clearly simple
    if simple_signals >= 1 and word_count < 30:
        return RoutingDecision(
            route=RouteType.FAST,
            model=cfg.MODEL_FAST,
            reason=f"simple factual query ({simple_signals} signals)",
            confidence=0.85,
        )

    # Default: fast (cheaper)
    return RoutingDecision(
        route=RouteType.FAST,
        model=cfg.MODEL_FAST,
        reason="default routing (short, no complex signals)",
        confidence=0.6,
    )


# ─── Router class ─────────────────────────────────────────────────────────────
class LLMRouter:
    """
    Drop-in streaming router — transparently selects the best model
    and yields SSE events with a routing header.
    """

    def __init__(self):
        self.client = get_async_client()
        self._stats: dict[str, int] = {r.value: 0 for r in RouteType}

    async def stream(
        self,
        messages: list[dict],
        requires_json: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """
        Yields SSE-formatted dicts. First event is always a routing event.
        Subsequent events are token/done/error events.
        """
        decision = classify_request(messages, requires_json)
        self._stats[decision.route.value] += 1

        # Announce routing decision
        yield {
            "event":      "routing",
            "model":      decision.model,
            "route":      decision.route.value,
            "reason":     decision.reason,
            "confidence": decision.confidence,
        }

        start = time.perf_counter()
        first_token_time: float | None = None
        token_count = 0

        try:
            kwargs = dict(
                model=decision.model,
                messages=messages,
                max_tokens=max_tokens or cfg.MAX_TOKENS,
                temperature=temperature or cfg.TEMPERATURE,
                stream=True,
            )
            if requires_json:
                kwargs["response_format"] = {"type": "json_object"}

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    yield {
                        "event":   "ttft",
                        "ms":      round((first_token_time - start) * 1000, 2),
                        "model":   decision.model,
                    }

                token_count += 1
                yield {"event": "token", "text": delta}

            total_ms = (time.perf_counter() - start) * 1000
            yield {
                "event":    "done",
                "model":    decision.model,
                "tokens":   token_count,
                "tps":      round(token_count / (total_ms / 1000), 1),
                "total_ms": round(total_ms, 1),
            }

        except Exception as e:
            yield {"event": "error", "msg": str(e), "model": decision.model}

    def routing_stats(self) -> dict:
        total = sum(self._stats.values()) or 1
        return {
            route: {"count": count, "pct": round(count / total * 100, 1)}
            for route, count in self._stats.items()
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────
async def demo():
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    cfg.validate()

    console.rule("[bold blue]🔀  Q3 — Intelligent Router Demo")

    router = LLMRouter()

    test_cases = [
        (False, [{"role": "user", "content": "What is the capital of Japan?"}]),
        (False, [{"role": "user", "content": "Explain how transformer attention mechanisms work and why multi-head attention is beneficial for capturing different types of relationships in text."}]),
        (True,  [{"role": "user", "content": "Extract all people, organizations, and dates mentioned: 'Elon Musk founded SpaceX in 2002 in Hawthorne, California.'"}]),
        (False, [{"role": "user", "content": "Write a Python class for a thread-safe LRU cache with TTL expiration."}]),
    ]

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", min_width=90)
    table.add_column("Prompt (truncated)",   min_width=35)
    table.add_column("Route",                justify="center")
    table.add_column("Model",                min_width=22)
    table.add_column("Reason",               min_width=25)
    table.add_column("TTFT (ms)",            justify="right")

    for requires_json, messages in test_cases:
        prompt_preview = messages[-1]["content"][:45] + "..."
        ttft = "—"
        model = "—"
        route = "—"
        reason = "—"

        console.print(f"\n▶ [dim]{prompt_preview}[/]")

        async for event in router.stream(messages, requires_json=requires_json, max_tokens=200):
            if event["event"] == "routing":
                model  = event["model"]
                route  = event["route"]
                reason = event["reason"]
                console.print(f"  → Routing to [bold cyan]{model}[/] ({reason})")
            elif event["event"] == "ttft":
                ttft = str(event["ms"])
            elif event["event"] == "done":
                console.print(f"  ✓ {event['tokens']} tokens @ {event['tps']} tok/s")
            elif event["event"] == "error":
                console.print(f"  ❌ {event['msg']}")

        color = {"fast": "green", "capable": "blue", "structured": "yellow"}.get(route, "white")
        table.add_row(
            prompt_preview,
            f"[{color}]{route}[/]",
            model,
            reason[:35],
            ttft,
        )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]📊 Routing Stats:[/]")
    for route, stats in router.routing_stats().items():
        console.print(f"  {route:12s}: {stats['count']:3d} requests  ({stats['pct']}%)")

    console.print("\n[bold green]✅  Router demo complete![/]\n")


if __name__ == "__main__":
    asyncio.run(demo())