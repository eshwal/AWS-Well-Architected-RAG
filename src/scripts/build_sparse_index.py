import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.service.factories.splitter_factory import SplitterFactory



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = "data/raw"
CACHE_PATH = "data/sparse_cache.json"

def save_cache(cache_data: list):
    """Saves the fresh BM25 cache json to disk for the FastAPI app to consume."""
    Path("data").mkdir(exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

def build_sparse_index():
    """Scans all documents, fully regenerates sparse JSON cache, and reloads BM25 index."""
    base_path = Path(DOCS_DIR)
    if not base_path.exists():
        logger.error(f"Directory {DOCS_DIR} does not exist.")
        return

    logger.info("[Sparse Builder] Scanning all AWS compliance documents...")
    files = list(base_path.rglob("*.md")) + list(base_path.rglob("*.txt"))
    
    if not files:
        logger.warning(f"No files found in {DOCS_DIR}.")
        return

    splitter = SplitterFactory.get_splitter("langchain_recursive", chunk_size=1000, chunk_overlap=200)
    new_cache_data = []

    for file_path in files:
        relative_name = str(file_path.relative_to(DOCS_DIR))
        content = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text_to_chunks(text=content,source_name=relative_name)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{relative_name}_ch_{idx}"
            new_cache_data.append({
                "id" : chunk_id,
                "page_content":chunk.page_content,
                "metadata":{"source": chunk.metadata["source"]}
            })

    logger.info(f"[Sparse Builder] Extracted {len(new_cache_data)} total chunks across {len(files)} files.")
    
    # 1. Save to disk so FastAPI can load it later
    save_cache(new_cache_data)

    logger.info("[Sparse Builder] BM25 cache successfully rebuilt and loaded.")

if __name__ == "__main__":
    build_sparse_index()