"""
RAG Engine - Core retrieval-augmented generation logic.

Architecture:
1. Document Processing: Load and chunk documents
2. Embedding: Convert chunks to vectors via sentence-transformers
3. Storage: FAISS vector index for fast similarity search
4. Retrieval: Top-k semantic search on query
5. Generation: LLM call with retrieved context + strict grounding prompt
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE     = 512     # characters per chunk
CHUNK_OVERLAP  = 64      # overlap between consecutive chunks
TOP_K          = 5       # retrieved chunks per query
MIN_SCORE      = 0.25    # minimum cosine similarity to include a source
EMBED_MODEL    = "all-MiniLM-L6-v2"  # fast, 384-dim, good quality


# ── Document Loader ───────────────────────────────────────────────────────────
def load_text(content: bytes, file_type: str) -> str:
    """Extract plain text from uploaded bytes."""
    if file_type in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")
    elif file_type == ".json":
        data = json.loads(content)
        # Flatten JSON to readable text
        return json.dumps(data, indent=2)
    elif file_type == ".pdf":
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries where possible."""
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to break at sentence boundary
        boundary = max(
            text.rfind(". ", start, end),
            text.rfind("\n", start, end),
            text.rfind("? ", start, end),
            text.rfind("! ", start, end),
        )
        if boundary > start + chunk_size // 2:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return [c for c in chunks if len(c.strip()) > 20]


# ── RAG Engine ────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, index_dir: str = "./rag_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        
        # FAISS index (cosine similarity via inner product on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        
        # Metadata parallel to FAISS vectors
        self.chunks: List[str] = []            # raw text of each chunk
        self.chunk_meta: List[Dict] = []       # {doc_name, chunk_idx}
        self.documents: Dict[str, int] = {}    # doc_name -> chunk count
        
        self._load_index()
        self._setup_llm()
    
    # ── LLM Setup ─────────────────────────────────────────────────────────
    def _setup_llm(self):
        """Configure LLM provider. Falls back gracefully."""
        self.llm_provider = "none"
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.llm_provider = "anthropic"
                logger.info("LLM provider: Anthropic Claude")
                return
            except ImportError:
                logger.warning("anthropic package not installed")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
                self.llm_provider = "openai"
                logger.info("LLM provider: OpenAI")
                return
            except ImportError:
                logger.warning("openai package not installed")
        
        logger.warning("No LLM API key found. Responses will be retrieval-only summaries.")
    
    # ── Index Persistence ─────────────────────────────────────────────────
    def _load_index(self):
        meta_path = self.index_dir / "meta.json"
        faiss_path = self.index_dir / "index.faiss"
        
        if meta_path.exists() and faiss_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.chunks = meta["chunks"]
            self.chunk_meta = meta["chunk_meta"]
            self.documents = meta["documents"]
            self.index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded index: {len(self.chunks)} chunks from {len(self.documents)} docs")
    
    def _save_index(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.index_dir / "meta.json"
        faiss_path = self.index_dir / "index.faiss"
        with open(meta_path, "w") as f:
            json.dump({"chunks": self.chunks, "chunk_meta": self.chunk_meta, "documents": self.documents}, f)
        faiss.write_index(self.index, str(faiss_path))
    
    # ── Document Management ───────────────────────────────────────────────
    def add_document(self, content: bytes, filename: str, file_type: str) -> int:
        """Process and index a document. Returns number of chunks added."""
        text = load_text(content, file_type)
        chunks = chunk_text(text)
        
        if not chunks:
            raise ValueError("Document produced no indexable content")
        
        # Remove existing version if re-uploading
        if filename in self.documents:
            self.remove_document(filename)
        
        # Embed and normalize for cosine similarity
        embeddings = self.embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.chunk_meta.extend({"doc_name": filename, "chunk_idx": i} for i, _ in enumerate(chunks))
        self.documents[filename] = len(chunks)
        
        self._save_index()
        logger.info(f"Indexed '{filename}': {len(chunks)} chunks")
        return len(chunks)
    
    def remove_document(self, doc_name: str) -> bool:
        if doc_name not in self.documents:
            return False
        
        # Identify chunks to keep
        keep = [i for i, m in enumerate(self.chunk_meta) if m["doc_name"] != doc_name]
        
        if keep:
            kept_chunks = [self.chunks[i] for i in keep]
            kept_meta = [self.chunk_meta[i] for i in keep]
            kept_embeddings = self.embedder.encode(kept_chunks, normalize_embeddings=True)
            kept_embeddings = np.array(kept_embeddings, dtype="float32")
            
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(kept_embeddings)
            self.chunks = kept_chunks
            self.chunk_meta = kept_meta
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunks = []
            self.chunk_meta = []
        
        del self.documents[doc_name]
        self._save_index()
        return True
    
    def list_documents(self) -> List[Dict]:
        return [{"name": name, "chunks": count} for name, count in self.documents.items()]
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def get_chunk_count(self) -> int:
        return len(self.chunks)
    
    def get_provider_info(self) -> str:
        return self.llm_provider
    
    # ── Retrieval ─────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)
        
        results = []
        seen_docs = {}  # Deduplicate by doc to show diverse sources
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < MIN_SCORE:
                continue
            meta = self.chunk_meta[idx]
            doc = meta["doc_name"]
            
            # Allow max 2 chunks per document for diversity
            if seen_docs.get(doc, 0) >= 2:
                continue
            seen_docs[doc] = seen_docs.get(doc, 0) + 1
            
            results.append({
                "text": self.chunks[idx],
                "doc_name": doc,
                "chunk_idx": meta["chunk_idx"],
                "score": round(float(score), 4)
            })
        
        return results
    
    # ── Generation ────────────────────────────────────────────────────────
    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> Tuple[str, str]:
        """Build system and user prompts for grounded generation."""
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n[Source {i} – {chunk['doc_name']}]\n{chunk['text']}\n"
        
        system = """You are a precise, helpful assistant that answers questions STRICTLY based on the provided source documents.

RULES (non-negotiable):
1. Use ONLY information from the provided sources. Do not use outside knowledge.
2. If the sources do not contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
3. Cite which source(s) your answer is based on using [Source N] notation.
4. Do not speculate, infer, or hallucinate facts not present in the sources.
5. Keep answers concise and factual."""
        
        user = f"""Sources:
{context_text}

Question: {query}

Answer based only on the sources above:"""
        
        return system, user
    
    def _generate_with_anthropic(self, system: str, user: str) -> str:
        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text
    
    def _generate_with_openai(self, system: str, user: str) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def _generate_fallback(self, query: str, chunks: List[Dict]) -> str:
        """No LLM available — return most relevant chunk as answer."""
        if not chunks:
            return "No documents are indexed yet. Please upload documents first."
        best = chunks[0]
        return (f"[No LLM configured — showing best matching excerpt from '{best['doc_name']}']\n\n"
                f"{best['text']}\n\n"
                f"To enable AI-generated answers, set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment.")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve → generate → return answer + sources."""
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(question)
        
        if not chunks:
            if self.index.ntotal == 0:
                answer = "No documents have been indexed yet. Please upload some documents first using the Upload button."
            else:
                answer = "I couldn't find relevant information in the indexed documents to answer your question."
            return {"answer": answer, "sources": []}
        
        # Step 2: Generate grounded answer
        system, user_prompt = self._build_prompt(question, chunks)
        
        try:
            if self.llm_provider == "anthropic":
                answer = self._generate_with_anthropic(system, user_prompt)
            elif self.llm_provider == "openai":
                answer = self._generate_with_openai(system, user_prompt)
            else:
                answer = self._generate_fallback(question, chunks)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = self._generate_fallback(question, chunks)
        
        # Step 3: Return answer + source metadata (exclude raw text for frontend)
        sources = [{"doc_name": c["doc_name"], "score": c["score"], "preview": c["text"][:200] + "..."} for c in chunks]
        
        return {"answer": answer, "sources": sources}
