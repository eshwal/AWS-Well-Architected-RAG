from pydantic import BaseModel,Field
from typing import Any

class Document(BaseModel):
    page_content:str
    metadata:dict[str,Any] = Field(default_factory=dict)


class IngestionChunk(Document):
    id:str


class RetrievalChunk(BaseModel):
    document: IngestionChunk
    score: float