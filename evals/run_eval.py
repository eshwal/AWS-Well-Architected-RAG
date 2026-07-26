from argparse import ArgumentParser
from evals.profiles import PROFILES
from evals.schema import load_goldens
from evals.invokers import ServiceInvoker,SkippedIntent
from evals.post_checks import source_overlap
from evals.eval import run as run_judge
from evals.reporting import aggregate,print_table
import asyncio
import datetime
from pathlib import Path
import json

from src.service.sparse_index_loader import load_sparse_index_from_cache

def load_previous_rows(previous_report_path: str | None) -> dict[str, dict]:
    """Rows from a previous run, keyed by id, reused if they already have a usable answer."""
    if not previous_report_path or not Path(previous_report_path).exists():
        return {}
    prev = json.loads(Path(previous_report_path).read_text())
    return {r["id"]: r for r in prev.get("rows", []) if r.get("answer", "").strip() and r.get("contexts")}



async def main():
    ap = ArgumentParser(description="Running Deepeval Evaluation using cli commands.")

    # Adding arguments 
    ap.add_argument(
         "--profile",
         required= True,
         choices=list(PROFILES.keys()),
         help = "To run eval with particular feature"

    )

    ap.add_argument(
        "--questions",
        default="eval/seed_ques.yaml",
        help="Path to seed questions YAML",
        
    )

    ap.add_argument(
        "--filter",
        default=None,
        help="Only run goldens with demonstrates_feature == FILTER + baseline",
    )
        
    
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/<timestamp>_<profile>.json)",
        
    )
    ap.add_argument("--resume-from", default=None,
                     help="Path to a previous run's output JSON — reuse its rows instead of "
                          "re-invoking/re-scoring, only fills in what's missing/errored")

    args = ap.parse_args()
    flags = PROFILES[args.profile]
    goldens = load_goldens(args.questions)

    if args.filter:
        goldens = [
            g
            for g in goldens if g.demonstrates_feature in (args.filter,"baseline")
        ]
    
    load_sparse_index_from_cache()

    invoker = ServiceInvoker()

    previous_rows = load_previous_rows(args.resume_from)
    rows = []
    skipped = []
    remaining_goldens = [g for g in goldens if g.id not in previous_rows]
    print(f"[invoke] {len(previous_rows)} reused from previous report, {len(remaining_goldens)} to invoke.")

    for idx,g in enumerate(remaining_goldens):
        print(f"Invoking Service: {idx+1}/{len(remaining_goldens)}")
        try:
            resp = await invoker.invoke(g.question, flags, g.intent)
        except SkippedIntent as e:
            skipped.append({"id": g.id, "reason": str(e)})
            continue
        except Exception as e:
            skipped.append({"id": g.id, "reason": f"error: {e}"})
            continue
        # validate the response BEFORE trusting it enough to evaluate
        error_msg = resp.get("error","")
        if not resp.get("answer", "").strip() or not resp.get("chunks"):
            skipped.append({"id": g.id, "reason": error_msg or "empty answer or context"})
            continue

        rows.append(
            {
                "id": g.id,
                "demonstrates_feature": g.demonstrates_feature,
                "intent": g.intent,
                "question": g.question,
                "answer": resp["answer"],
                "contexts": resp["chunks"],
                "ground_truth": g.ground_truth,
                "actual_sources": resp["source"],
                "golden_sources": g.golden_sources,
                "query_type": g.query_type,
            }
        )

    rows.extend(previous_rows.values()) 
    metrics_by_id = run_judge(rows, previous_report_path=args.resume_from) if rows else {}
 
    for row in rows:
        row["deepeval_metrics"] = metrics_by_id.get(row["id"])  # None if judge never scored this row
        row["source_overlap"] = source_overlap(row["actual_sources"], row["golden_sources"])
 
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "profile": args.profile,
        "flags": flags,
        "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filter": args.filter,
        "rows": rows,
        "skipped": skipped,
        "aggregate": aggregate(metrics_by_id),
    }
 
    outpath = (
        Path(args.output) if args.output
        else Path(f"deepevals/output/{timestamp:%Y%m%dT%H%M%SZ}_{args.profile}.json")
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(payload, indent=2, default=str))
 
    print_table(rows)
    print(f"\nEvaluation output stored at {outpath}")



if __name__ == "__main__":
    asyncio.run(main())
