"""
RAG Chatbot Backend - FastAPI Application
"""
import os, json, uuid
from pathlib import Path
from typing import List, Optional
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

rag = RAGEngine()

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str

@app.get("/")
async def root():
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RAG Chatbot API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    session_id = request.session_id or str(uuid.uuid4())
    try:
        result = rag.query(request.message)
        return ChatResponse(answer=result["answer"], sources=result["sources"], session_id=session_id)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_types = {".txt", ".md", ".pdf", ".json"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported.")
    content = await file.read()
    try:
        chunks_added = rag.add_document(content=content, filename=file.filename, file_type=file_ext)
        return {"message": f"Document '{file.filename}' indexed successfully", "chunks_added": chunks_added, "total_documents": rag.get_document_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    return {"documents": rag.list_documents()}

@app.delete("/api/documents/{doc_name}")
async def delete_document(doc_name: str):
    success = rag.remove_document(doc_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found")
    return {"message": f"Document '{doc_name}' removed successfully"}

@app.get("/api/status")
async def status():
    return {"status": "operational", "documents_indexed": rag.get_document_count(), "total_chunks": rag.get_chunk_count(), "llm_provider": rag.get_provider_info()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
