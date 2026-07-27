from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging
import re
from src.service.interfaces.provider_interface import BaseEmbeddingEngine,BaseVectorDatabase
from src.service.factories.provider_factory import InfrastructureFactory
from src.service.hybrid import rank_fusion
from src.service.sparse_index_loader import get_sparse_index
from src.config import settings
from src.utils import retry_with_backoff
import asyncio
from langsmith import traceable
logger = logging.getLogger(__name__)

LLM_SEMAPHORE = asyncio.Semaphore(2)
EMBEDDINGS_PROVIDER = settings.EMBEDDINGS_PROVIDER
MODEL_PROVIDER = settings.MODEL_PROVIDER
VECTOR_DB_PROVIDER = settings.VECTOR_DB_PROVIDER


embedding_engine:BaseEmbeddingEngine = InfrastructureFactory.get_embedding_engine(EMBEDDINGS_PROVIDER)
embeddings = embedding_engine.get_native_embeddings()
model = InfrastructureFactory.get_model(MODEL_PROVIDER).get_llm()
vector_store:BaseVectorDatabase = InfrastructureFactory.get_vector_database(VECTOR_DB_PROVIDER,embedding_engine)

# rag.py — a small, standalone classifier, co-located with retrieval logic since it's reusable
_IDENTIFIER_PATTERN = re.compile(
    r'[a-z]+_[a-z]+_[a-z]+'      # snake_case tokens like x_amz_bedrock_agentcore_search
    r'|\b[A-Z]{2,}[A-Z]?\d{2,}\b'  # doc codes like AGENTSEC02, SUS01
)

def should_use_sparse_fallback(query: str) -> bool:
    """Heuristic: route to sparse when the query contains an exact identifier
    or doc code, where lexical match reliably beats semantic search."""
    return bool(_IDENTIFIER_PATTERN.search(query))

def get_prompt()->ChatPromptTemplate:
    ''' Returns the augmented prompt'''
    compliance_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AWS Compliance and Well-Architected Framework Advisor.\n"
            "Evaluate the query using only the provided reference context. "
            "Provide factual, precise architectural guidance based strictly on the context below. "
            "If the context does not contain enough information to address the query, state that explicitly.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}")
    ])
    return compliance_prompt

def get_chain():

    compliance_prompt = get_prompt()
    
    rag_chain = (
        compliance_prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

def format_docs(docs:list):
    return "\n\n".join(f'[Sources: {doc.document.metadata.get("source","Unknown")}]:\n{doc.document.page_content}' for doc in docs)

@traceable(name="sparse_retrieval")
@retry_with_backoff(max_attempts=3,base_delay=2)
async def sparse_context(user_query:str,top_k:int=5):
    '''Create context for sparse search'''
    sparse_index = get_sparse_index()
    sparse_docs = sparse_index.sparse_search(
        user_query,
        top_k
    )

    return sparse_docs, format_docs(sparse_docs)

@traceable(name="dense_retrieval")
@retry_with_backoff(max_attempts=3,base_delay=2)
async def dense_context(user_query:str,top_k:int=5):
    '''Create context for vector search'''
    dense_docs = vector_store.vector_search(user_query,top_k)

    return dense_docs,format_docs(dense_docs)

@traceable(name="hybrid_retrieval")
@retry_with_backoff(max_attempts=3,base_delay=2)
async def hybrid_context(user_query:str,sparse_top_k:int=10,dense_top_k:int=10,top_k:int=5,rrf_const:int=60):
    try:
        retrieved_docs = vector_store.vector_search(user_query,top_k=dense_top_k)
        sparse_index = get_sparse_index()
        sparse_docs = sparse_index.sparse_search(user_query,top_k=sparse_top_k)
        fuse_docs = rank_fusion([retrieved_docs,sparse_docs],rrf_const=rrf_const,top_k=top_k)

        return fuse_docs,format_docs(fuse_docs)
    except Exception as e:
        logger.error(f'Could not create hybrid context:{str(e)}')
        raise

@traceable(name="generation")
@retry_with_backoff(max_attempts=3, base_delay=2)
async def generate_answer(context, question):
    rag_chain = get_chain()
    async with LLM_SEMAPHORE:
        return await rag_chain.ainvoke({"context": context, "question": question})

async def query_compliance_platform(user_query: str,flags) -> dict:
    """Executes non-blocking vector search and response generation via pure LCEL."""
    
    # 1. Fetch source documents explicitly first so we can return metadata
    answer = ""
    sources = []
    chunks = []

    mode = flags.get("search_mode","dense")
    top_k = flags.get("top_k",5)

    try:
        if mode == "hybrid":
            rrf_const = flags.get("rrf_const",60)
            docs, context = await hybrid_context(user_query,top_k=top_k,rrf_const=rrf_const)
        
        elif mode =="dense":
            docs, context = await dense_context(user_query,top_k=top_k)

        elif mode=="sparse":
            docs,context = await sparse_context(user_query,top_k=top_k)  
    except Exception as e:
        logger.error(f"Retrieval failed for mode={mode}: {e}")
        # docs/context stay empty — same graceful-degradation choice as generation below,
        # rather than crashing the whole request
        return {
            "query": user_query,
            "answer": "",
            "source": [],
            "referenced_metadata": [],
            "chunks": [],
            "error": f"retrieval_failed: {e}",
        }

    sources = [
            {
                "id": doc.document.id,
                "source": doc.document.metadata.get("source","Unknown"),
                "snippet": doc.document.page_content[:200]+"..."
    
            }
            for doc in docs
        ]
    chunks = [doc.document.page_content for doc in docs]
    try:
        
        answer = await generate_answer(context=context,question=user_query)

    except Exception as e:
        logger.error(f'RAG chain failed due to: {str(e)}')
        return {
            "query": user_query,
            "answer": "",
            "source": list({d.document.metadata.get("source", "Unknown") for d in docs}),
            "referenced_metadata":sources,
            "chunks": chunks,
            "error": f"generation_failed: {e}",
        }

    return {
        "query": user_query,
        "answer": answer,
        "source": list({s["source"] for s in sources}),
        "referenced_metadata": sources,
        "chunks": chunks
    }