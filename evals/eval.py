import os
import time
from pathlib import Path
import json

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import LiteLLMModel
from deepeval.evaluate.configs import(
    AsyncConfig,
    ErrorConfig,
)
from src.config import settings

os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE
os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = settings.DEEPEVAL_RETRY_MAX_ATTEMPTS
os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = settings.DEEPEVAL_RETRY_CAP_SECONDS
os.environ["DEEPEVAL_RETRY_INITIAL_SECONDS"] = settings.DEEPEVAL_RETRY_INITIAL_SECONDS

llm = LiteLLMModel(
    api_key=settings.GRADER_API_KEY,
    model=settings.GRADER_MODEL,
    base_url=settings.GRADER_BASE_URL,
    generation_kwargs={"response_format": {"type": "json_object"}},
)
 
MAX_CONTEXT_CHUNKS = 4  # cap chunks sent to judge — reduces token usage on tight quotas


# ==============================================================================
# 📊 DATASET PROCESSING & EXECUTION
# ==============================================================================

def build_dataset(rows: list[dict]) -> list[LLMTestCase]:
    test_cases = []
    for r in rows:
        test_case=LLMTestCase(input=r["question"],
               actual_output= r["answer"],
               expected_output=r["ground_truth"],
               retrieval_context=r["contexts"][:MAX_CONTEXT_CHUNKS]
        )
        test_cases.append(test_case)
    return test_cases

def extract_metrics(test_result) -> list[dict]:
    """Convert deepeval's MetricData objects into plain JSON-safe dicts."""
    return [
        {
            "name": getattr(md, "name", None),
            "score": getattr(md, "score", None),
            "success": getattr(md, "success", None),
            "reason": getattr(md, "reason", None),
            "error": getattr(md, "error", None),
        }
        for md in test_result.metrics_data
    ]

def _has_error(metric_list: list[dict] | None) -> bool:
    if not metric_list:
        return True
    return any(m.get("error") or m.get("score") is None for m in metric_list)
def get_metrics() -> list:
    # async_mode=False everywhere: keeps calls to the judge model sequential,
    # which matters on tight free-tier rate limits. Flip to True if move
    # to a paid tier with real headroom.
    return [
        FaithfulnessMetric(model=llm, async_mode=False),
        AnswerRelevancyMetric(model=llm, async_mode=False),
        ContextualPrecisionMetric(model=llm, async_mode=False),
        ContextualRecallMetric(model=llm, async_mode=False),
        ContextualRelevancyMetric(model=llm, async_mode=False),
    ]

def run(rows: list[dict], previous_report_path: str | None = None, sleep_between: int = 20) -> dict[str, list[dict]]:
    """
    Scores each row against deepeval's 5 metrics, one row per evaluate() call.
    If previous_report_path points at an earlier run's output JSON, rows that
    already scored cleanly there are reused instead of re-scored — saves quota
    on reruns after hitting a rate/quota wall partway through.
    Returns {row_id: [metric_dict, ...]}.
    """
    if not rows:
        return {}
 
    metrics_by_id: dict[str, list[dict]] = {}
    if previous_report_path and Path(previous_report_path).exists():
        prev = json.loads(Path(previous_report_path).read_text())
        for row in prev.get("rows", []):
            dm = row.get("deepeval_metrics")
            if not _has_error(dm):
                metrics_by_id[row["id"]] = dm
 
    remaining = [r for r in rows if r["id"] not in metrics_by_id]
    print(f"[judge] {len(metrics_by_id)} reused from previous report, {len(remaining)} to score.")
 
    metrics = get_metrics()
    async_config = AsyncConfig(run_async=False)
    error_config = ErrorConfig(ignore_errors=True)  # lets a run complete with partial/errored rows rather than crashing
 
    for i, r in enumerate(remaining):
        tc = build_dataset([r])[0]
        print(f"[judge] [{i+1}/{len(remaining)}] scoring {r['id']}")
        result = evaluate(test_cases=[tc], metrics=metrics, async_config=async_config, error_config=error_config)
        metrics_by_id[r["id"]] = extract_metrics(result.test_results[0])
        if i < len(remaining) - 1:
            time.sleep(sleep_between)
 
    return metrics_by_id

