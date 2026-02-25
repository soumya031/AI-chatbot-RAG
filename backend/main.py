import os
import json
import uuid
from pathlib import Path
from typing import List, Optional
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocMind RAG Chatbot API",
    version="1.0.1",
    description="A domain-specific RAG chatbot that answers questions strictly from uploaded documents"
)

# Initialize RAG engine
rag = RAGEngine()

# Frontend path
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# CORS middleware - more restrictive for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Restrict to local development
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error for {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="The user's question or message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for conversation continuity")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The AI-generated answer or retrieval result")
    sources: List[dict] = Field(..., description="List of source documents with relevance scores")
    session_id: str = Field(..., description="Session identifier for this conversation")

class DocumentUploadResponse(BaseModel):
    message: str = Field(..., description="Success message")
    chunks_added: int = Field(..., description="Number of text chunks added to the index")
    total_documents: int = Field(..., description="Total number of documents now indexed")

class StatusResponse(BaseModel):
    status: str = Field(..., description="API status")
    documents_indexed: int = Field(..., description="Number of documents indexed")
    total_chunks: int = Field(..., description="Total number of text chunks")
    llm_provider: str = Field(..., description="LLM provider currently configured")

@app.get("/")
async def root():
    """Serve the frontend or return API information."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "DocMind RAG Chatbot API",
        "version": "1.0.1",
        "docs": "/docs",
        "status": "/api/status"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": str(uuid.uuid4())}

@app.get("/api/version")
async def get_version():
    """Get API version information."""
    return {
        "version": "1.0.1",
        "name": "DocMind RAG Chatbot",
        "description": "Domain-specific RAG chatbot with document indexing"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return an answer based on indexed documents."""
    logger.info(f"Processing chat request: session_id={request.session_id}, message_length={len(request.message)}")

    if not request.message.strip():
        logger.warning("Empty message received")
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = rag.query(request.message)
        logger.info(f"Query processed successfully for session {session_id}")
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error processing query for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document for the RAG system."""
    logger.info(f"Processing document upload: {file.filename}, size={file.size} bytes")

    allowed_types = {".txt", ".md", ".pdf", ".json"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        logger.warning(f"Unsupported file type: {file_ext}")
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_types)}")

    if file.size > 10 * 1024 * 1024:  # 10MB limit
        logger.warning(f"File too large: {file.size} bytes")
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        chunks_added = rag.add_document(content=content, filename=file.filename, file_type=file_ext)
        logger.info(f"Document '{file.filename}' indexed successfully: {chunks_added} chunks added")

        return DocumentUploadResponse(
            message=f"Document '{file.filename}' indexed successfully",
            chunks_added=chunks_added,
            total_documents=rag.get_document_count()
        )
    except ValueError as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    return {"documents": rag.list_documents()}

@app.delete("/api/documents/{doc_name}")
async def delete_document(doc_name: str):
    success = rag.remove_document(doc_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found")
    return {"message": f"Document '{doc_name}' removed successfully"}

@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Get the current status of the RAG system."""
    return StatusResponse(
        status="operational",
        documents_indexed=rag.get_document_count(),
        total_chunks=rag.get_chunk_count(),
        llm_provider=rag.get_provider_info()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
