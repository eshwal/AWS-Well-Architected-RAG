from src.service.interfaces.splitter_interface import BaseSplitter
from src.service.implementations.splitter_implementation import LangchainRecursiveSplitter

class SplitterFactory:
    @staticmethod
    def get_splitter(strategy_name: str = "langchain_recursive", **kwargs) -> BaseSplitter:
        if strategy_name == "langchain_recursive":
            return LangchainRecursiveSplitter(
                chunk_size=kwargs.get("chunk_size", 1000),
                chunk_overlap=kwargs.get("chunk_overlap", 200)
            )
        else:
            raise ValueError(f"Unknown text splitter strategy: {strategy_name}")


