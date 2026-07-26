from abc import ABC,abstractmethod

from src.models.document import Document

class BaseSplitter(ABC):
    '''Abstract class for splitter'''
    @abstractmethod
    def split_text_to_chunks(self,text: str, source_name: str) -> list[Document]:
        """Takes raw text and returns a unified list of Document structures."""
        pass