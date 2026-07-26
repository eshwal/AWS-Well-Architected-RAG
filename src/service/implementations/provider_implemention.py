from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document as LCDoc
from typing_extensions import Any
from src.models.document import IngestionChunk,RetrievalChunk
from src.service.interfaces.provider_interface import BaseLLMProvider,BaseEmbeddingEngine,BaseVectorDatabase
from src.config import settings



class LangchainMistralLLM(BaseLLMProvider):
    def __init__(self):
        self.llm = ChatMistralAI(
            api_key = settings.MODEL_API_KEY,
            model = settings.GENERATION_MODEL
        )

    def get_llm(self):
        return self.llm

class LangChainMistralEmbedding(BaseEmbeddingEngine):
    def __init__(self):
        self.model = MistralAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY
        )
        
    def get_native_embeddings(self):
        return self.model

class LangChainPineconeDatabase(BaseVectorDatabase):
    def __init__(self, embedding_engine: BaseEmbeddingEngine):
        native_embeddings = embedding_engine.get_native_embeddings()
        self.store = PineconeVectorStore(
            index_name=settings.INDEX_NAME, 
            embedding=native_embeddings
        )

    def insert_documents(self, documents: list[IngestionChunk]) -> None:
        lc_docs = [
            LCDoc(
            id = doc.id,
            page_content= doc.page_content,
            metadata = doc.metadata
        )
        for doc in documents
        ]

        chunk_ids = [doc.id for doc in documents]
        self.store.add_documents(documents=lc_docs, ids=chunk_ids)

    def delete_documents(self, chunk_ids: list[str]) -> None:
        try:
            self.store.delete(ids=chunk_ids)
        except Exception as e:
            print(f'Exception occured while deleting {chunk_ids} with exception {str(e)}')

    def delete_by_source(self, source):
        try:
            self.store.delete(filter={"source":source})
        except Exception as e:
            print(f'Exception occured while deleting {source} with exception {str(e)}')

    def vector_search(self, query: str, top_k: int,filters: dict[str, Any] | None = None) -> list[RetrievalChunk]:
        namespace = filters.get("namespace") if filters else None

        metadata_filter = {
            k: v
            for k, v in (filters or {}).items()
            if k != "namespace"
        }
        if not metadata_filter:
            metadata_filter = None
        raw_resp = self.store.similarity_search_with_score(query, k=top_k,filter=metadata_filter,namespace=namespace)
        return [
            RetrievalChunk(
                document=IngestionChunk(id=doc.id,page_content=doc.page_content, metadata=doc.metadata),
                score= score
            )
            for doc, score in raw_resp
        ]