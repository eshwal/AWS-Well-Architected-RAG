from rank_bm25 import BM25Okapi
import re
import numpy as np

from src.models.document import RetrievalChunk,IngestionChunk

class SparseVectorIndex:
    ''' Class for creating and managing sparse index'''
    def __init__(self):
        self.corpus = []
        self.bm25 = None

    def _tokenize(self,text:str)->list[str]:
        return re.findall(r'\w+',text.lower())
    
    def fit(self,documents:list[IngestionChunk]):
        self.corpus = documents
        if not self.corpus:
            return
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.corpus]

        self.bm25 = BM25Okapi(tokenized_docs)

    def sparse_search(self,query:str,top_k:int=2)-> list[RetrievalChunk]:
        if not self.corpus or self.bm25 is None:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        rank_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in rank_indices:
            if scores[idx]>0:
                doc = self.corpus[idx]
                chunk = RetrievalChunk(
                    document=doc,
                    score = float(scores[idx])
                )
                results.append(chunk)
        return results
        
