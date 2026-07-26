from abc import abstractmethod,ABC
from typing_extensions import Any
from src.models.document import IngestionChunk,RetrievalChunk


class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm():
        """ Returns the llm as per provider """
        pass

class BaseEmbeddingEngine(ABC):
    @abstractmethod
    def get_native_embeddings(self):
        """Returns the underlying wrapper object needed by database engines."""
        pass

class BaseVectorDatabase(ABC):
    @abstractmethod
    def insert_documents(self, documents: list[IngestionChunk]) -> None:
        """Uploads text chunk documents into storage."""
        pass

    @abstractmethod
    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Purges target documents using explicit keys."""
        pass
    
    @abstractmethod
    def delete_by_source(self,source:str)->None:
        """Purges target document using source"""
    @abstractmethod
    def vector_search(self, query: str, top_k: int,filters: dict[str, Any] | None = None,) -> list[RetrievalChunk]:
        """Executes similarity query and returns a standardized result payload."""
        pass

