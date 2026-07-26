from pathlib import Path
from pydantic import BaseModel,Field
from typing_extensions import Literal
import yaml


INTENT = Literal["rag"]
FEATURE = Literal[
    "baseline",
    "sparse",
    "dense",
    "hybrid",
    "rerank",
    "hyde",
    "crag",
    "self_rag",
]
QUERY_TYPE = Literal[
    "informal_realistic",
    "semantic_paraphrase",
    "exact_keyword",
    "multi-hop",
    "unanswerable"
]

class Golden(BaseModel):
    id: str = Field(..., pattern=r"^test-\d{3}$")
    question: str = Field(..., min_length=1)
    intent: INTENT
    query_type: QUERY_TYPE
    ground_truth: str | None=None
    golden_sources: list[str] = Field(..., min_length=0)
    demonstrates_feature: FEATURE
    notes: str


def load_goldens(path: str|Path):
    path = Path(path)

    text = path.read_text("utf-8")

    raw = yaml.safe_load(text)

    if not isinstance(raw, list):
        raise ValueError(f"Expected YAML root to be a list, got {type(raw).__name__}")

    goldens = [Golden(**golden) for golden in raw]

    ids = [g.id for g in goldens]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"Duplicate golden IDs found: {duplicates}")
    print(len(goldens))
    return goldens





if __name__ == "__main__":
    load_goldens("eval/manual_test.yaml")
