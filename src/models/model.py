from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question",
    )

class SourceMetadata(BaseModel):
    id: str
    source: str
    snippet: str

class ChatResponse(BaseModel):
    query: str
    answer: str = Field(default="")
    source: list[str] = Field(default_factory=list)
    referenced_metadata: list[SourceMetadata] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    error: str | None = None


    




