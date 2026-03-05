"""
q3_structured/structured_stream.py
────────────────────────────────────
Q3 — Grammar-constrained structured output with streaming.

Since Groq doesn't expose a grammar constraint API directly, we use:
  1. Strict JSON-mode prompting  (reliable for Llama 3 / Mixtral)
  2. Streaming partial JSON      (token by token, parsed progressively)
  3. Schema validation on output (jsonschema)

Run:
    cd llm_streaming
    python -m q3_structured.structured_stream
"""
import sys, json, asyncio, time, re
from pathlib import Path
from typing import AsyncIterator, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from shared.config import cfg
from shared.groq_client import get_async_client

console = Console()


# ─── JSON Schemas ──────────────────────────────────────────────────────────────
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "people":        {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
        "locations":     {"type": "array", "items": {"type": "string"}},
        "dates":         {"type": "array", "items": {"type": "string"}},
        "key_facts":     {"type": "array", "items": {"type": "string"}},
    },
    "required": ["people", "organizations", "locations", "dates", "key_facts"],
    "additionalProperties": False,
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment":   {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
        "score":       {"type": "number",  "minimum": -1.0, "maximum": 1.0},
        "confidence":  {"type": "number",  "minimum": 0.0,  "maximum": 1.0},
        "reasoning":   {"type": "string"},
        "key_phrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentiment", "score", "confidence", "reasoning", "key_phrases"],
}

CODE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score":    {"type": "integer", "minimum": 1, "maximum": 10},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity":    {"type": "string", "enum": ["critical", "major", "minor", "info"]},
                    "line":        {"type": "integer"},
                    "description": {"type": "string"},
                    "suggestion":  {"type": "string"},
                },
                "required": ["severity", "description", "suggestion"],
            },
        },
        "strengths":     {"type": "array", "items": {"type": "string"}},
        "summary":       {"type": "string"},
    },
    "required": ["overall_score", "issues", "strengths", "summary"],
}


# ─── Core: Streaming JSON generator ──────────────────────────────────────────
async def stream_structured_json(
    user_prompt: str,
    schema: dict,
    model: str | None = None,
    max_tokens: int = 1024,
) -> AsyncIterator[dict]:
    """
    Yields dicts:
      {"event": "token",    "text": "{", "accumulated": "{"}
      {"event": "partial",  "valid_so_far": True, "accumulated": "..."}
      {"event": "done",     "result": {...}, "valid": True, "ttft_ms": 210.3}
      {"event": "error",    "msg": "..."}
    """
    _model = model or cfg.MODEL_STRUCTURED
    client = get_async_client()

    # Build a strict system prompt that forces JSON output matching our schema
    system = (
        "You are a precise data extraction assistant. "
        "You MUST respond with a single valid JSON object that exactly matches the provided schema. "
        "Do NOT include any text before or after the JSON. "
        "Do NOT include markdown code fences. "
        "Output ONLY the raw JSON object.\n\n"
        f"Required JSON Schema:\n{json.dumps(schema, indent=2)}"
    )

    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user_prompt},
    ]

    start = time.perf_counter()
    first_token_time: float | None = None
    accumulated = ""

    try:
        stream = await client.chat.completions.create(
            model=_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,    # Low temp for consistent structured output
            stream=True,
            response_format={"type": "json_object"},   # Groq JSON mode
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - start) * 1000
                yield {"event": "ttft", "ms": round(ttft_ms, 2)}

            accumulated += delta
            yield {"event": "token", "text": delta, "accumulated": accumulated}

        # Parse and validate
        total_ms = (time.perf_counter() - start) * 1000

        # Strip any accidental markdown fences
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", accumulated.strip())

        try:
            result = json.loads(clean)
            jsonschema.validate(instance=result, schema=schema)
            yield {
                "event":    "done",
                "result":   result,
                "valid":    True,
                "total_ms": round(total_ms, 1),
                "ttft_ms":  round((first_token_time - start) * 1000, 1) if first_token_time else 0,
            }
        except json.JSONDecodeError as e:
            yield {"event": "error", "msg": f"JSON parse error: {e}", "raw": accumulated}
        except jsonschema.ValidationError as e:
            yield {"event": "error", "msg": f"Schema validation failed: {e.message}", "partial": accumulated}

    except Exception as e:
        yield {"event": "error", "msg": str(e)}


# ─── High-level helpers ────────────────────────────────────────────────────────
async def extract_entities(text: str) -> dict | None:
    """Extract named entities from text. Returns parsed dict or None."""
    prompt = f"Extract all named entities from the following text:\n\n{text}"
    result = None
    async for event in stream_structured_json(prompt, ENTITY_SCHEMA):
        if event["event"] == "done":
            result = event["result"]
        elif event["event"] == "error":
            console.print(f"[red]Error: {event['msg']}[/]")
    return result


async def analyze_sentiment(text: str) -> dict | None:
    """Analyze sentiment with structured output."""
    prompt = f"Analyze the sentiment of the following text:\n\n{text}"
    result = None
    async for event in stream_structured_json(prompt, SENTIMENT_SCHEMA):
        if event["event"] == "done":
            result = event["result"]
        elif event["event"] == "error":
            console.print(f"[red]Error: {event['msg']}[/]")
    return result


async def review_code(code: str) -> dict | None:
    """Code review with structured output."""
    prompt = f"Review the following Python code:\n\n```python\n{code}\n```"
    result = None
    async for event in stream_structured_json(prompt, CODE_REVIEW_SCHEMA):
        if event["event"] == "done":
            result = event["result"]
        elif event["event"] == "error":
            console.print(f"[red]Error: {event['msg']}[/]")
    return result


# ─── Demo runner ──────────────────────────────────────────────────────────────
async def demo():
    cfg.validate()
    console.rule("[bold green]🧩  Q3 — Structured Output Streaming Demo")

    # ── Demo 1: Entity Extraction ──────────────────────────────────────────
    console.print("\n[bold cyan]Demo 1: Entity Extraction[/]")
    text = (
        "Apple CEO Tim Cook announced yesterday in Cupertino, California that "
        "the company will invest $1 billion in AI infrastructure by Q4 2025. "
        "The announcement was welcomed by Satya Nadella at Microsoft and "
        "Jensen Huang of NVIDIA. The news broke on January 15, 2025."
    )
    console.print(f"[dim]Input:[/] {text}\n")

    tokens_received = 0
    async for event in stream_structured_json(
        f"Extract entities from: {text}", ENTITY_SCHEMA
    ):
        if event["event"] == "ttft":
            console.print(f"⚡ TTFT: [green]{event['ms']} ms[/]")
        elif event["event"] == "token":
            tokens_received += 1
            # Show progress dots every 10 tokens
            if tokens_received % 10 == 0:
                console.print(".", end="", highlight=False)
        elif event["event"] == "done":
            console.print(f"\n✅ [green]Valid JSON![/] ({event['total_ms']} ms total)")
            console.print(Syntax(json.dumps(event["result"], indent=2), "json"))
        elif event["event"] == "error":
            console.print(f"\n❌ [red]{event['msg']}[/]")

    # ── Demo 2: Sentiment Analysis ─────────────────────────────────────────
    console.print("\n[bold cyan]Demo 2: Sentiment Analysis[/]")
    review = (
        "This library is absolutely fantastic! The documentation is clear, "
        "the API is intuitive, and performance is outstanding. Minor gripe: "
        "the error messages could be a bit more descriptive."
    )
    console.print(f"[dim]Input:[/] {review}\n")

    result = await analyze_sentiment(review)
    if result:
        console.print(Syntax(json.dumps(result, indent=2), "json"))

    # ── Demo 3: Code Review ────────────────────────────────────────────────
    console.print("\n[bold cyan]Demo 3: Code Review[/]")
    code = '''
def find_user(users, id):
    for i in range(len(users)):
        if users[i]["id"] == id:
            return users[i]
    return None

password = "admin123"  # todo: move this
'''
    console.print(Panel(Syntax(code.strip(), "python"), title="Code to review"))

    result = await review_code(code)
    if result:
        console.print(Syntax(json.dumps(result, indent=2), "json"))

    console.print("\n[bold green]✅  Q3 Demo complete![/]\n")


if __name__ == "__main__":
    asyncio.run(demo())