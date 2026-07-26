from src.service.interfaces.provider_interface import(
    BaseLLMProvider,
    BaseEmbeddingEngine,
    BaseVectorDatabase
)
from src.service.implementations.provider_implemention import (
    LangChainMistralEmbedding,
    LangchainMistralLLM,
    LangChainPineconeDatabase)

class InfrastructureFactory:
    @staticmethod
    def get_model(provider: str) -> BaseLLMProvider:
        if provider == "langchain_mistral":
            return LangchainMistralLLM()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    @staticmethod
    def get_embedding_engine(provider: str) -> BaseEmbeddingEngine:
        if provider == "langchain_mistral":
            return LangChainMistralEmbedding()
        else:
            raise ValueError(f"Unknown embedding provider : {provider}")

    @staticmethod
    def get_vector_database(provider: str, embedding_engine: BaseEmbeddingEngine) -> BaseVectorDatabase:
        if provider == "langchain_pinecone":
            return LangChainPineconeDatabase(embedding_engine=embedding_engine)
        else:
            raise ValueError(f"Unknown vector storage allocation key: {provider}")