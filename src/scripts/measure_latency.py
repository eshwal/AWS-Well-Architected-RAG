"""
One-off latency benchmark. Run manually to get p50/p95 numbers for the README 
"""
import asyncio
import functools
import json
import statistics
import time
from pathlib import Path

from src.service.sparse_index_loader import load_sparse_index_from_cache
from src.service.rag import sparse_context, generate_answer  

LOG_FILE = Path("latency_results.json")

SAMPLE_QUESTIONS = [
    # pick ~10-12 across your query_type categories
    "How do I prevent an agent from getting overwhelmed by tool overload?",
    "agent keeps calling same tool over and over how to stop that",
    "What does x_amz_bedrock_agentcore_search do?",
    "How can i measure instance utilization goals for sustainability?",
    "What's AWS's refund policy if I cancel a reserved EC2 instance early?",
    "How does Amazon Bedrock AgentCore Memory handle service unreachability?",
    "mcp vs a2a whats the diff",
    "What fallback behaviors should I define when a capability is disabled?",
    "zero trust AWS internal services explain like im new to this",
    "What format is required for maintaining log and data?",
]


def timed(stage: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log(stage, elapsed_ms)
            return result
        return wrapper
    return decorator


def _log(stage: str, elapsed_ms: float):
    entry = {"stage": stage, "elapsed_ms": elapsed_ms, "ts": time.time()}
    data = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    data.append(entry)
    LOG_FILE.write_text(json.dumps(data, indent=2))


@timed("retrieval_sparse")
async def timed_retrieval(question: str):
    return await sparse_context(question)


@timed("generation")
async def timed_generation(context: str, question: str):
    return await generate_answer(context, question)


async def run():
    load_sparse_index_from_cache()

    for q in SAMPLE_QUESTIONS:
        start = time.perf_counter()
        docs, context = await timed_retrieval(q)
        answer = await timed_generation(context, q)
        end_to_end_ms = (time.perf_counter() - start) * 1000
        _log("end_to_end", end_to_end_ms)
        print(f"done: {q[:50]}...")


def report():
    data = json.loads(LOG_FILE.read_text())
    by_stage = {}
    for entry in data:
        by_stage.setdefault(entry["stage"], []).append(entry["elapsed_ms"])

    print("\n=== Latency (n per stage, ms) ===")
    for stage, values in by_stage.items():
        values.sort()
        p50 = statistics.median(values)
        p95 = values[int(len(values) * 0.95)] if len(values) > 1 else values[0]
        print(f"{stage:<20} n={len(values):<3} p50={p50:.0f}ms  p95={p95:.0f}ms")


if __name__ == "__main__":
    if LOG_FILE.exists():
        LOG_FILE.unlink()  # fresh run each time
    asyncio.run(run())
    report()