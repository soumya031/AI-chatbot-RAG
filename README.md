# DocMind — Domain-Specific RAG Chatbot

A chatbot that answers questions **strictly from your uploaded documents** using Retrieval-Augmented Generation (RAG). No hallucinations — every answer is grounded in source content with citations.

---

## Architecture

---

## Preview

<p align="center">
  <img src="preview/docmind-ui.png" alt="DocMind UI" width="900"/>
</p>

The DocMind chatbot features a clean, modern interface with:

- Left sidebar for document management (upload, view, and delete documents)
- Main chat area for asking questions and receiving answers
- Source citations showing which documents contributed to each answer
- Real-time status indicators showing system health and active LLM provider
- Responsive design that works on both desktop and mobile devices

---

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DocMind RAG Pipeline                        │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Document │───▶│  Chunker     │───▶│  Sentence Transformer    │  │
│  │ Loader   │    │ (512 chars,  │    │  (all-MiniLM-L6-v2)      │  │
│  │.txt .md  │    │  64 overlap) │    │  384-dim embeddings      │  │
│  │.pdf .json│    └──────────────┘    └────────────┬─────────────┘  │
│  └──────────┘                                     │                │
│                                                   ▼                │
│                                        ┌──────────────────────┐    │
│                                        │   FAISS Vector Index │    │
│                                        │  (IndexFlatIP +      │    │
│                                        │   cosine similarity) │    │
│                                        └──────────┬───────────┘    │
│                                                   │ top-5 chunks   │
│  ┌───────────────────────────────────────────────▼─────────────┐  │
│  │                       Query Pipeline                         │  │
│  │                                                              │  │
│  │  User Query ──▶ Embed ──▶ FAISS Search ──▶ Retrieved Chunks │  │
│  │                                                   │         │  │
│  │                           ┌──────────────────────▼──────┐   │  │
│  │                           │  Grounded Prompt Assembly   │   │  │
│  │                           │  (system: strict grounding  │   │  │
│  │                           │   rules + source context)   │   │  │
│  │                           └──────────────┬──────────────┘   │  │
│  │                                          │                  │  │
│  │                                    ┌─────▼──────┐           │  │
│  │                                    │  LLM (API) │           │  │
│  │                                    │  Claude /  │           │  │
│  │                                    │  OpenAI /  │           │  │
│  │                                    │  Fallback  │           │  │
│  │                                    └─────┬──────┘           │  │
│  │                                          │                  │  │
│  │                               Answer + Source Citations     │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Document Loading** | Custom Python | Parse .txt, .md, .pdf, .json |
| **Chunking** | Custom splitter | 512-char chunks, 64-char overlap, sentence-aware |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) | 384-dim semantic vectors |
| **Vector Store** | `faiss-cpu` (IndexFlatIP) | Fast cosine similarity search |
| **LLM** | Anthropic Claude / OpenAI GPT / Fallback | Grounded answer generation |
| **API** | FastAPI | REST backend |
| **Frontend** | Vanilla HTML/CSS/JS | Single-file chat interface |

---

## Setup

### Prerequisites
- Python 3.10+
- An Anthropic or OpenAI API key (optional but recommended)

### Installation

```bash
git clone 
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Configuration

```bash
# Set your LLM API key (at least one)
export ANTHROPIC_API_KEY="your-key-here"   # Preferred: Claude
# OR
export OPENAI_API_KEY="your-key-here"      # Alternative: GPT-4o-mini
```

> **No API key?** The system still works — it returns the best-matching raw excerpt instead of a generated answer.

### Running the Application

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

### Quick Test with Sample Documents

```bash
# Optional: index sample docs via the API
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@sample_docs/ai_fundamentals.txt"

curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@sample_docs/company_handbook.txt"
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `POST` | `/api/chat` | Submit a query |
| `POST` | `/api/documents/upload` | Upload & index a document |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{name}` | Remove a document |
| `GET` | `/api/status` | System health check |

### Chat Request/Response

```json
// POST /api/chat
{ "message": "What is RAG?", "session_id": "optional-uuid" }

// Response
{
  "answer": "RAG (Retrieval-Augmented Generation) is... [Source 1]",
  "sources": [
    { "doc_name": "ai_fundamentals.txt", "score": 0.89, "preview": "RAG is an AI technique..." }
  ],
  "session_id": "abc-123"
}
```

---

## Design Decisions

### Why FAISS over a hosted vector DB?
FAISS runs in-process with zero infrastructure overhead — perfect for a self-contained deployment. The index is persisted to disk and loaded on startup. For production scale (millions of documents), a switch to Pinecone, Weaviate, or pgvector would be appropriate.

### Why `all-MiniLM-L6-v2`?
It's fast (384 dims vs 1536 for OpenAI ada), runs entirely locally (no API cost), and achieves strong retrieval quality on general text. For highly specialized domains, fine-tuning or domain-specific models could improve recall.

### Why 512-char chunks with 64-char overlap?
This balances context richness vs. retrieval precision. Larger chunks embed more context but dilute the semantic signal; smaller chunks are precise but may lose surrounding context. Overlap ensures boundary sentences aren't split between chunks.

### Hallucination prevention
The system prompt explicitly instructs the LLM to:
1. Use **only** information from provided sources
2. Cite which source each claim comes from
3. Say "I don't know" if sources are insufficient
4. Never speculate or infer beyond stated facts

### Index persistence
The FAISS index and metadata are saved to `./rag_index/` as `index.faiss` and `meta.json`, enabling the knowledge base to survive server restarts.

---

## Potential Improvements

With additional time, the following enhancements would meaningfully improve the system:

**Retrieval Quality**
- Hybrid search (BM25 keyword + dense vector) using Reciprocal Rank Fusion for better recall on exact-match queries
- Re-ranking retrieved chunks with a cross-encoder model before LLM generation
- Parent-child chunking: index small chunks but pass larger parent chunks as context

**System Features**
- Multi-turn conversation with chat history stored per session
- Streaming responses via Server-Sent Events for lower perceived latency
- Background indexing queue for large document batches
- OCR support for scanned PDFs using Tesseract

**Production Hardening**
- Authentication (API keys / OAuth) for multi-user deployments
- Rate limiting and request queuing
- Async embedding generation using a worker pool
- Swap FAISS for Weaviate/pgvector for distributed or cloud deployments
- Monitoring dashboard (Grafana + LangSmith for trace logging)

**Quality & Testing**
- RAGAS evaluation framework for automated retrieval + answer quality measurement
- Unit tests for chunking edge cases and embedding consistency
- End-to-end integration tests with known Q&A pairs
