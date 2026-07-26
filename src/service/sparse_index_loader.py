import json
import logging
from pathlib import Path
from typing import Optional
from src.service.sparse_vector_service import SparseVectorIndex
from src.models.document import IngestionChunk

logger = logging.getLogger(__name__)

# Single responsibility: Holds the active in-memory BM25 matrix
GLOBAL_SPARSE_INDEX: Optional[SparseVectorIndex] = None

def load_sparse_index_from_cache(cache_path_str: str = "data/sparse_cache.json"):
    """Loads cache file and performs an atomic in-memory swap for BM25."""
    global GLOBAL_SPARSE_INDEX
    cache_path = Path(cache_path_str)
    
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents = [IngestionChunk(page_content=d["page_content"],metadata=d["metadata"],id=d["id"]) for d in data]
                if documents:
                    new_sparse = SparseVectorIndex()
                    new_sparse.fit(documents)
                    GLOBAL_SPARSE_INDEX = new_sparse
                    logger.info(f"[Sparse Index] Loaded {len(documents)} document chunks into BM25 memory.")
                    return
        except Exception as e:
            logger.error(f"[Sparse Index] Failed loading cache: {e}")
            
    GLOBAL_SPARSE_INDEX = None



def get_sparse_index():
    return  GLOBAL_SPARSE_INDEX
