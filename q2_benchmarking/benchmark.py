"""
q2_benchmarking/benchmark.py
─────────────────────────────
Q2 — Throughput & latency benchmarking against Groq models.
Simulates what you'd do when comparing vLLM vs baseline — but
using Groq as the backend (same metrics, real streaming).

Run:
    cd llm_streaming
    python -m q2_benchmarking.benchmark
"""
import sys, time, asyncio, statistics, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

from shared.config import cfg
from shared.groq_client import get_async_client

console = Console()

# ── Benchmark prompts (varied lengths & types) ────────────────────────────────
PROMPTS = [
    # Short / factual
    ("short_factual",    "What is the capital of France?"),
    ("short_factual",    "How many planets are in the solar system?"),
    # Medium / analytical
    ("medium_analytical","Explain the difference between TCP and UDP in 3 sentences."),
    ("medium_analytical","What are the main trade-offs between SQL and NoSQL databases?"),
    # Long / generative
    ("long_generative",  "Write a Python function that implements a binary search tree with insert, search, and delete methods. Include docstrings."),
    ("long_generative",  "Explain how transformer attention mechanisms work, including the query-key-value computation and why multi-head attention is useful."),
]


@dataclass
class RequestResult:
    model: str
    prompt_type: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tps: float
    success: bool
    error: str = ""


@dataclass
class BenchmarkSummary:
    model: str
    results: list[RequestResult] = field(default_factory=list)

    def success_rate(self) -> float:
        if not self.results: return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results) * 100

    def ttft_stats(self) -> dict:
        vals = [r.ttft_ms for r in self.results if r.success]
        if not vals: return {}
        return {
            "p50": round(statistics.median(vals), 1),
            "p90": round(sorted(vals)[int(len(vals) * 0.9)], 1),
            "p99": round(sorted(vals)[int(len(vals) * 0.99) if len(vals) >= 100 else -1], 1),
            "mean": round(statistics.mean(vals), 1),
        }

    def throughput_stats(self) -> dict:
        vals = [r.tps for r in self.results if r.success]
        if not vals: return {}
        return {
            "mean_tps": round(statistics.mean(vals), 1),
            "total_tokens": sum(r.output_tokens for r in self.results),
        }


# ── Single request benchmark ──────────────────────────────────────────────────
async def _benchmark_single(
    client,
    model: str,
    prompt_type: str,
    prompt: str,
    max_tokens: int = 256,
) -> RequestResult:
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    first_token_time: float | None = None
    token_count = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1  # approx 1 token per chunk from Groq

        total_ms = (time.perf_counter() - start) * 1000
        ttft_ms  = ((first_token_time or start) - start) * 1000
        tps      = token_count / (total_ms / 1000) if total_ms > 0 else 0

        return RequestResult(
            model=model,
            prompt_type=prompt_type,
            prompt_tokens=len(prompt.split()),   # rough estimate
            output_tokens=token_count,
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            tps=round(tps, 1),
            success=True,
        )

    except Exception as e:
        total_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            model=model,
            prompt_type=prompt_type,
            prompt_tokens=0,
            output_tokens=0,
            ttft_ms=0,
            total_ms=round(total_ms, 2),
            tps=0,
            success=False,
            error=str(e),
        )


# ── Concurrent load test ───────────────────────────────────────────────────────
async def run_concurrent_benchmark(
    model: str,
    concurrency: int,
    iterations: int,
    max_tokens: int = 256,
) -> BenchmarkSummary:
    """
    Runs `iterations` requests with `concurrency` simultaneous requests.
    Returns a BenchmarkSummary with all results.
    """
    client = get_async_client()
    summary = BenchmarkSummary(model=model)
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(pt, p):
        async with semaphore:
            return await _benchmark_single(client, model, pt, p, max_tokens)

    # Cycle through prompts
    tasks = []
    for i in range(iterations):
        pt, p = PROMPTS[i % len(PROMPTS)]
        tasks.append(_bounded(pt, p))

    with Progress(
        SpinnerColumn(),
        TextColumn(f"  [bold blue]{model}[/] concurrency={concurrency}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            result = await coro
            summary.results.append(result)
            progress.advance(task_id)

    return summary


# ── Report rendering ──────────────────────────────────────────────────────────
def print_report(summaries: list[BenchmarkSummary]):
    console.print("\n")
    console.rule("[bold cyan]📊  Benchmark Results")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Model",           style="bold white",  min_width=22)
    table.add_column("Requests",        justify="right")
    table.add_column("Success %",       justify="right")
    table.add_column("TTFT P50 (ms)",   justify="right")
    table.add_column("TTFT P90 (ms)",   justify="right")
    table.add_column("Mean Tok/s",      justify="right")
    table.add_column("Total Tokens",    justify="right")

    for s in summaries:
        ttft  = s.ttft_stats()
        tput  = s.throughput_stats()
        sr    = s.success_rate()
        sr_col = f"[green]{sr:.1f}%[/]" if sr >= 99 else f"[yellow]{sr:.1f}%[/]"

        table.add_row(
            s.model,
            str(len(s.results)),
            sr_col,
            str(ttft.get("p50", "—")),
            str(ttft.get("p90", "—")),
            str(tput.get("mean_tps", "—")),
            str(tput.get("total_tokens", "—")),
        )

    console.print(table)

    # Save JSON report
    report_path = Path(__file__).parent / "benchmark_report.json"
    report = [
        {
            "model": s.model,
            "total_requests": len(s.results),
            "success_rate": s.success_rate(),
            "ttft": s.ttft_stats(),
            "throughput": s.throughput_stats(),
        }
        for s in summaries
    ]
    report_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n💾  Report saved → [cyan]{report_path}[/]\n")


# ── Main entry point ──────────────────────────────────────────────────────────
async def main():
    cfg.validate()
    console.rule("[bold blue]🔬  Q2 — LLM Benchmark Suite (Groq)")
    console.print(f"   Fast model    : [cyan]{cfg.MODEL_FAST}[/]")
    console.print(f"   Capable model : [cyan]{cfg.MODEL_CAPABLE}[/]")
    console.print()

    # Define benchmark matrix: (model, concurrency, iterations)
    matrix = [
        (cfg.MODEL_FAST,    1,  10),
        (cfg.MODEL_FAST,    5,  20),
        (cfg.MODEL_CAPABLE, 1,  10),
        (cfg.MODEL_CAPABLE, 3,  12),
    ]

    summaries = []
    for model, concurrency, iters in matrix:
        console.print(f"\n▶  Running: [bold]{model}[/] | concurrency={concurrency} | n={iters}")
        summary = await run_concurrent_benchmark(model, concurrency, iters)
        summaries.append(summary)

    print_report(summaries)


if __name__ == "__main__":
    asyncio.run(main())