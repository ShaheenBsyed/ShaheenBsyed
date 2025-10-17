from __future__ import annotations

from typing import List, Optional
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.services.retriever import FaissRetriever
from app.services.generator import LocalGenerator
from app.config import settings

app = FastAPI(title="Custom Dataset AI Chatbot")


class ChatRequest(BaseModel):
    messages: List[dict]
    top_k: int = 5


retriever: Optional[FaissRetriever] = None
llm: Optional[LocalGenerator] = None


@app.on_event("startup")
async def startup_event():
    global retriever, llm
    retriever = FaissRetriever()
    if settings.generation.provider == "transformers":
        llm = LocalGenerator()
    else:
        raise RuntimeError("Only local transformers provider is implemented in this template")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    index_file = Path(settings.base_dir) / "web" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found; build the web UI")
    return FileResponse(str(index_file))


@app.post("/chat")
async def chat(req: ChatRequest):
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # last user message
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    question = user_messages[-1].get("content", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    hits = retriever.search(question, k=req.top_k)
    contexts = [h.content for h in hits]

    async def event_generator():
        for token in llm.stream_answer(question, contexts):
            yield {"data": json.dumps({"event": "token", "data": token})}
        yield {"data": json.dumps({"event": "done", "data": ""})}

    return EventSourceResponse(event_generator())
