import json
import os
import asyncio
from pathlib import Path
from src.service.rag import query_compliance_platform 

def load_eval_dataset(filepath="eval/eval_dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate_pipeline(eval_dir:str|Path="eval/eval_dataset.json"):
    dataset = load_eval_dataset(eval_dir)
    total = len(dataset)
    sparse_hits = 0
    hybrid_hits, dense_hits = 0, 0
    recovered, broken = 0, 0
    

    print(f"=== Starting A/B Evaluation across {total} test queries ===\n")
    print(f"{'Mode':<8} | {'Status':<6} | {'Query'}")
    print("-" * 75)

    
    for item in dataset:

        # Code for confusion matrix
        query = item["query"]
        expected_source = os.path.normpath(item["expected_source"])

        # 1. Test Dense Mode
        dense_res = await query_compliance_platform(query, flags={"search_mode": "dense"})
        # 2. Test Sparse Mode
        sparse_res = await query_compliance_platform(query, flags={"search_mode": "sparse"})
        # 3. Test Hybrid Mode (with over-fetching default)
        hybrid_res = await query_compliance_platform(query, flags={"search_mode": "hybrid"})
        
        d = any(expected_source in os.path.normpath(s["source"]) for s in dense_res["referenced_metadata"])
        if d:
            dense_hits += 1
        sparse_pass = any(expected_source in os.path.normpath(src["source"]) for src in sparse_res["referenced_metadata"])
        if sparse_pass:
            sparse_hits += 1
        h = any(expected_source in os.path.normpath(s["source"]) for s in hybrid_res["referenced_metadata"])
        if h:
            hybrid_hits += 1

        if h and not d: recovered += 1
        if d and not h: broken += 1

        print(f"Dense  | {'PASS' if d else 'FAIL':<6} | {query}")
        print(f"Sparse | {'PASS' if sparse_pass else 'FAIL':<6} | {query}")
        print(f"Hybrid | {'PASS' if h else 'FAIL':<6} | {query}")
        print("-" * 75)

    print("\n=== Final Evaluation Summary ===")
    print(f"  - Dense Only Hit Rate:   {(dense_hits / total) * 100:.1f}% ({dense_hits}/{total})")
    print(f"  - Sparse Only Hit Rate:  {(sparse_hits / total) * 100:.1f}% ({sparse_hits}/{total})")
    print(f"  - Hybrid Search Hit Rate:{(hybrid_hits / total) * 100:.1f}% ({hybrid_hits}/{total})")
    print(f"Hybrid:  Recovered: {recovered} | Broken: {broken}")
    
    if hybrid_hits >= dense_hits and hybrid_hits >= sparse_hits:
        print("\nVerdict: Hybrid search is successfully capturing both exact keywords and concepts. Implementing it was an effective architectural choice.")
    else:
        print("\nVerdict: Standalone search outperformed hybrid. You may want to check your RRF fusion parameters or increase your over-fetch depth.")

if __name__ == "__main__":
    eval_dir = "docs/decisions/hybrid-vs-dense-investigation/eval_dataset.json"
    asyncio.run(evaluate_pipeline(eval_dir))