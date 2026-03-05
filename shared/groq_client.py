"""
shared/groq_client.py
─────────────────────
Thin wrapper around the Groq client.
Groq's API is OpenAI-compatible, so we use the openai SDK
pointed at Groq's base URL — works with stream=True out of the box.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI, OpenAI
from shared.config import cfg


def get_async_client() -> AsyncOpenAI:
    """Return an async Groq-compatible OpenAI client."""
    return AsyncOpenAI(
        api_key=cfg.GROQ_API_KEY,
        base_url=cfg.GROQ_API_BASE,
    )


def get_sync_client() -> OpenAI:
    """Return a sync Groq-compatible OpenAI client (for benchmarks/scripts)."""
    return OpenAI(
        api_key=cfg.GROQ_API_KEY,
        base_url=cfg.GROQ_API_BASE,
    )