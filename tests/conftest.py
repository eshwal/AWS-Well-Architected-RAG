import pytest
from src.models.document import RetrievalChunk,IngestionChunk
import os

os.environ.setdefault("INDEX_NAME","test_index")
os.environ.setdefault("EMBEDDINGS_PROVIDER","langchain_mistral")
os.environ.setdefault("MODEL_PROVIDER","langchain_mistral")
os.environ.setdefault("VECTOR_DB_PROVIDER","langchain_pinecone")
os.environ.setdefault("EMBEDDING_API_KEY","")
os.environ.setdefault("EMBEDDING_MODEL","")
os.environ.setdefault("VECTOR_DB_API_KEY","")
os.environ.setdefault("MODEL_API_KEY","")
os.environ.setdefault("GENERATION_MODEL","test")
os.environ.setdefault("GRADER_MODEL", "test")
os.environ.setdefault("GRADER_API_KEY","test")
os.environ.setdefault("GRADER_BASE_URL","test")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE","1200")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS","3")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS","60")     # max wait between retries
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS","5")   # initial backoff
os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
os.environ.setdefault("LANGCHAIN_API_KEY","test")
os.environ.setdefault("LANGCHAIN_PROJECT","test")

@pytest.fixture()
def asynio_backend():
    return "asyncio"

@pytest.fixture
def make_doc():
    def _make(doc_id="1", content="some content", source="test.md"):
        return RetrievalChunk(
            document=IngestionChunk(id=doc_id, page_content=content, metadata={"source": source}),
            score=1.0,
        )
    return _make
