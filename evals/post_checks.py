def source_overlap(actual: list[str], golden: list[str]) -> dict:
    import os

    try:
        actual_set = {os.path.normpath(s) for s in actual}
        golden_set = {os.path.normpath(s) for s in golden}
        overlap = actual_set & golden_set

        return {
            "overlap_pct": round(len(overlap) / max(len(golden_set), 1), 3),
            "matched": sorted(overlap),
            "missed": sorted(golden_set - actual_set),
        }
    except Exception as e:
        print(f'Error:{str(e)}')
        return {}