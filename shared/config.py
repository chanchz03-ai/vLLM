"""
shared/config.py
────────────────
Central config loaded from .env — imported by all quarters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of which sub-dir you run from)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Config:
    # ── Groq credentials ───────────────────────────────────────────────────
    GROQ_API_KEY: str        = os.getenv("GROQ_API_KEY", "")
    GROQ_API_BASE: str       = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    # ── Models ─────────────────────────────────────────────────────────────
    MODEL_FAST: str          = os.getenv("GROQ_MODEL_FAST",       "llama3-8b-8192")
    MODEL_CAPABLE: str       = os.getenv("GROQ_MODEL_CAPABLE",    "llama-3.3-70b-versatile")
    MODEL_STRUCTURED: str    = os.getenv("GROQ_MODEL_STRUCTURED",  "llama3-70b-8192")

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str                = os.getenv("APP_HOST", "0.0.0.0")
    PORT: int                = int(os.getenv("APP_PORT", "8000"))
    ENV: str                 = os.getenv("APP_ENV", "development")

    # ── Streaming ──────────────────────────────────────────────────────────
    STREAM_TIMEOUT: float    = float(os.getenv("STREAM_TIMEOUT_SECONDS", "60"))
    MAX_TOKENS: int          = int(os.getenv("MAX_TOKENS_DEFAULT", "1024"))
    TEMPERATURE: float       = float(os.getenv("TEMPERATURE_DEFAULT", "0.7"))

    # ── Observability ──────────────────────────────────────────────────────
    ENABLE_METRICS: bool     = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    PROMETHEUS_PORT: int     = int(os.getenv("PROMETHEUS_PORT", "9090"))

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY or cls.GROQ_API_KEY == "your_groq_api_key_here":
            raise EnvironmentError(
                "❌  GROQ_API_KEY is not set.\n"
                "    Edit the .env file and add your key from https://console.groq.com"
            )
        print(f"✅  Config loaded | env={cls.ENV} | model_fast={cls.MODEL_FAST}")


cfg = Config()