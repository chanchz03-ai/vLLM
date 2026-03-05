"""
run.py
───────
Unified CLI entry point for all quarters.

Usage:
    python run.py q1          # Start Q1 streaming chat server
    python run.py q2          # Run Q2 benchmarks
    python run.py q3          # Run Q3 structured output demo
    python run.py q3-router   # Run Q3 router demo
    python run.py q4          # Start Q4 production server
    python run.py demo        # Quick end-to-end demo (no server needed)
"""
import sys, asyncio
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║    ⚡  LLM Streaming Skill-Up 2025  |  Groq Backend      ║
╠══════════════════════════════════════════════════════════╣
║  q1          → Streaming chat server (port 8000)         ║
║  q2          → Benchmark suite (models × concurrency)    ║
║  q3          → Structured output streaming demo          ║
║  q3-router   → Multi-model intelligent router demo       ║
║  q4          → Production server + observability         ║
║  demo        → Quick single-turn streaming demo          ║
╚══════════════════════════════════════════════════════════╝
""")


async def quick_demo():
    """A quick single-turn streaming demo — no server needed."""
    from shared.config import cfg
    from q1_foundations.streaming import stream_chat
    from rich.console import Console

    cfg.validate()
    console = Console()
    console.rule("[bold blue]⚡  Quick Streaming Demo")

    prompt = "Explain what Server-Sent Events are and why they're ideal for LLM streaming. Keep it under 100 words."
    console.print(f"\n[dim]Prompt:[/] {prompt}\n")
    console.print("[bold cyan]Response:[/] ", end="")

    ttft_shown = False
    async for event in stream_chat(
        messages=[{"role": "user", "content": prompt}],
        model=cfg.MODEL_FAST,
        max_tokens=150,
    ):
        import json
        data = json.loads(event[6:])   # strip "data: "
        if data["event"] == "ttft" and not ttft_shown:
            console.print(f"[dim]\n⚡ TTFT: {data['ms']}ms\n[/]", end="")
            ttft_shown = True
        elif data["event"] == "token":
            print(data["text"], end="", flush=True)
        elif data["event"] == "done":
            console.print(f"\n\n[dim]✓ {data['tokens']} tokens @ {data['tps']} tok/s | {data['total_ms']}ms total[/]")


def main():
    print_banner()
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if cmd == "q1":
        import uvicorn
        from shared.config import cfg
        cfg.validate()
        uvicorn.run(
            "q1_foundations.server:app",
            host=cfg.HOST, port=cfg.PORT, reload=True,
        )

    elif cmd == "q2":
        from q2_benchmarking.benchmark import main as bench_main
        asyncio.run(bench_main())

    elif cmd == "q3":
        from q3_structured.structured_stream import demo
        asyncio.run(demo())

    elif cmd == "q3-router":
        from q3_structured.router import demo
        asyncio.run(demo())

    elif cmd == "q4":
        import uvicorn
        from shared.config import cfg
        cfg.validate()
        print("📊  Metrics → http://localhost:8000/metrics")
        print("🏥  Health  → http://localhost:8000/health")
        print("📈  Stats   → http://localhost:8000/stats\n")
        uvicorn.run(
            "q4_observability.production_server:app",
            host=cfg.HOST, port=cfg.PORT, reload=False,
        )

    elif cmd == "demo":
        asyncio.run(quick_demo())

    else:
        print(f"Unknown command: {cmd}")
        print("Run: python run.py [q1|q2|q3|q3-router|q4|demo]")
        sys.exit(1)


if __name__ == "__main__":
    main()