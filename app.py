from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from search_service import GraphRAGLocalSearcher

app = FastAPI(title="GraphRAG Neo4j API", version="1.0.0")
searcher = GraphRAGLocalSearcher()


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="检索条数")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    return searcher.answer_question(payload.question, top_k=payload.top_k)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    return StreamingResponse(
        searcher.stream_answer(payload.question, top_k=payload.top_k),
        media_type="text/event-stream",
    )
