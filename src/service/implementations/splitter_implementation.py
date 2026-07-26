from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDoc

from src.models.document import Document
from src.service.interfaces.splitter_interface import BaseSplitter

class LangchainRecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def split_text_to_chunks(self,text, source_name)->list[Document]:
        lc_doc = LCDoc(
            page_content= text,
            metadata={"source":source_name}
        )

        split_docs = self.splitter.split_documents([lc_doc])
        return [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in split_docs]