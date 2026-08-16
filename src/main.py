from fastapi import FastAPI,Depends
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

from src.auth import verify_api_key
from src.service.sparse_index_loader import load_sparse_index_from_cache
from src.models.model import ChatRequest,ChatResponse
from src.service.rag import query_compliance_platform,should_use_sparse_fallback


@asynccontextmanager
async def lifespan(app:FastAPI):
    load_sparse_index_from_cache()
    yield

app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app,endpoint="/metrics")

@app.post("/query",response_model=ChatResponse,dependencies=[Depends(verify_api_key)])
async def query(req:ChatRequest):
    question = req.question

    mode = "sparse" if should_use_sparse_fallback(question) else "dense"

    flags = {"search_mode": mode, "top_k": 4}
    resp = await query_compliance_platform(question, flags)
    return resp

    

