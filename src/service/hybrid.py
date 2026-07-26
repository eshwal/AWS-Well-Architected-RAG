from src.models.document import IngestionChunk,RetrievalChunk


def rank_fusion(combined_chunks:list[list[RetrievalChunk]],rrf_const:int=60,top_k:int=4)->list[RetrievalChunk]:
    '''Combines on the basis of reciprocal rank'''
    rrf_score = {}
    chunk_meta = {}
    for chunks in combined_chunks:
        for rank ,chunk in enumerate(chunks):
            key = chunk.document.id
            rrf_score[key]=rrf_score.get(key,0.0)+(1/(rrf_const+1+rank))
            if key not in chunk_meta:
                chunk_meta[key]={"page_content":chunk.document.page_content,"metadata":chunk.document.metadata}
    
    
    return [
        RetrievalChunk(
            document=IngestionChunk(
                id=chunk_id,
                page_content=chunk_meta[chunk_id]["page_content"],
                metadata=chunk_meta[chunk_id]["metadata"]
                ),
            score=score
        )
        for chunk_id,score in sorted(rrf_score.items(),key=lambda x:x[1],reverse=True)
    ][:top_k]
    