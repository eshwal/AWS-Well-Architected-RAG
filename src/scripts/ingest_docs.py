import argparse
import logging
from pathlib import Path
from pinecone import Pinecone,ServerlessSpec

from src.models.document import IngestionChunk
from src.service.interfaces.provider_interface import BaseVectorDatabase
from src.service.factories.provider_factory import InfrastructureFactory
from src.service.factories.splitter_factory import SplitterFactory
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = "data/raw"
BATCH_SIZE = 100

def batch_upsert(vector_db:BaseVectorDatabase, doc_objects: list[IngestionChunk], batch_size: int = BATCH_SIZE):
    import time, random, json
    from pathlib import Path

    CHECKPOINT = Path("data/ingest_checkpoint.json")
    done_batches = set(json.loads(CHECKPOINT.read_text())) if CHECKPOINT.exists() else set()
    total = len(doc_objects)

    for i in range(0, total, batch_size):
        if i in done_batches:
            continue
        batch_docs = doc_objects[i : i + batch_size]
        logger.info(f"[Batch Vector DB] Upserting items {i+1} to {min(i+batch_size, total)} of {total}...")
        attempt = 0
        while attempt < 5:
            try:
                vector_db.insert_documents(batch_docs)
                done_batches.add(i)
                CHECKPOINT.write_text(json.dumps(list(done_batches)))
                break
            except Exception as e:
                wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 2)
                logger.warning(f"Batch {i} failed ({e}), retrying in {wait:.0f}s [{attempt+1}/5]")
                time.sleep(wait)
                attempt += 1
        else:
            logger.error(f"Batch {i} failed after 5 attempts — stopping. Rerun the script to resume.")
            return

def run_full_sync(vector_db:BaseVectorDatabase, base_path: Path, splitter, reset_db: bool = False):
    if reset_db:
        logger.info("[Full Sync] Purging all vectors from Pinecone vector store...")
        try:
            vector_db.store.delete(delete_all=True)
        except Exception as e:
            logger.warning(f"Failed to clear Pinecone index: {e}")

    logger.info("[Full Sync] Scanning all AWS compliance documents for Dense Ingestion...")
    files = list(base_path.rglob("*.md"))
    
    if not files:
        logger.warning(f"No files found in {DOCS_DIR}.")
        return

    all_doc_objects = []

    for file_path in files:
        relative_name = str(file_path.relative_to(base_path))
        
        if not reset_db:
            vector_db.delete_by_source(relative_name)

        content = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text_to_chunks(content,relative_name)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{relative_name}_ch_{idx}"
            all_doc_objects.append(IngestionChunk(page_content=chunk.page_content, metadata=chunk.metadata,id=chunk_id))

    logger.info(f"[Full Sync] Extracted {len(all_doc_objects)} total chunks across {len(files)} files.")
    batch_upsert(vector_db, all_doc_objects, batch_size=BATCH_SIZE)

def run_single_update(vector_db, target_file: str, base_path: Path, splitter):
    file_path = base_path / target_file
    if not file_path.exists():
        logger.error(f"File {target_file} not found in {DOCS_DIR}.")
        return

    relative_name = str(file_path.relative_to(base_path))
    logger.info(f"[Targeted Dense Update] Processing file: {relative_name}")

    vector_db.delete_by_source(relative_name)

    content = file_path.read_text(encoding="utf-8")
    chunks = splitter.split_text_to_chunks(content,relative_name)
    
    doc_objects = [IngestionChunk(id=f"{relative_name}_ch_{idx}",page_content=c.page_content, metadata=c.metadata) for idx,c in enumerate(chunks)]

    batch_upsert(vector_db, doc_objects, batch_size=BATCH_SIZE)



def ensure_pinecone_index_exists(index_name: str, dimension: int = 1024):
    """Checks if the Pinecone index exists, and creates it if it doesn't."""
    pc = Pinecone(api_key=settings.VECTOR_DB_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info(f"[Pinecone] Index '{index_name}' not found. Creating it automatically...")
        pc.create_index(
            name=index_name,
            dimension=dimension,  # Match your Mistral embedding dimension (Mistral embed is usually 1024)
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Adjust to your Pinecone region if different
            )
        )
        logger.info(f"[Pinecone] Index '{index_name}' successfully created.")
    else:
        logger.info(f"[Pinecone] Index '{index_name}' already exists.")

def run_pipeline(mode: str, target_file: str = None, reset_db: bool = False):
    base_path = Path(DOCS_DIR)
    if not base_path.exists():
        logger.error(f"Directory {DOCS_DIR} does not exist.")
        return

    EMBEDDING_PROVIDER = settings.EMBEDDINGS_PROVIDER
    VECTOR_DB_PROVIDER = settings.VECTOR_DB_PROVIDER
    
    index_name = settings.INDEX_NAME
    
    # 1. AUTO-CREATE INDEX IF MISSING BEFORE PIPELINE RUNS
    ensure_pinecone_index_exists(index_name)

    embedding_provider = InfrastructureFactory.get_embedding_engine(EMBEDDING_PROVIDER)
    vector_db = InfrastructureFactory.get_vector_database(VECTOR_DB_PROVIDER, embedding_provider)
    splitter = SplitterFactory.get_splitter("langchain_recursive", chunk_size=1000, chunk_overlap=200)

    if mode == "full-sync":
        run_full_sync(vector_db, base_path, splitter, reset_db=reset_db)

    elif mode == "update" and target_file:
        run_single_update(vector_db, target_file, base_path, splitter)

    elif mode == "delete" and target_file:
        logger.info(f"[Targeted Dense Delete] Purging file: {target_file}")
        vector_db.delete_by_source(target_file)

    logger.info("[Dense Pipeline] Ingestion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Vector (Pinecone) Ingestion CLI")
    parser.add_argument("--mode", choices=["update", "delete", "full-sync"], default="full-sync")
    parser.add_argument("--file", type=str, help="Target filename (e.g., security_pillar.md)")
    parser.add_argument("--reset-db", action="store_true", help="Wipe Pinecone index before full sync")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, target_file=args.file, reset_db=args.reset_db)