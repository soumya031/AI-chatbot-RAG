import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_engine import chunk_text, load_text

def test_chunking():
    text = "Hello world. " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1, "Should produce multiple chunks"
    for c in chunks:
        assert len(c) > 0, "Chunks should not be empty"
    print(f"✓ Chunking: {len(chunks)} chunks from {len(text)} chars")

def test_load_text_txt():
    content = b"This is a test document.\nWith multiple lines."
    text = load_text(content, ".txt")
    assert "test document" in text
    print("✓ TXT loading")

def test_load_text_json():
    import json
    data = {"key": "value", "list": [1, 2, 3]}
    content = json.dumps(data).encode()
    text = load_text(content, ".json")
    assert "value" in text
    print("✓ JSON loading")

def test_rag_engine_basic():
    from rag_engine import RAGEngine
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = RAGEngine(index_dir=os.path.join(tmp_dir, "test_rag_index"))
    
    # Add a document
    doc_content = b"The capital of France is Paris. The Eiffel Tower is in Paris."
    chunks = engine.add_document(doc_content, "test_doc.txt", ".txt")
    assert chunks > 0, "Should index at least 1 chunk"
    print(f"✓ Document indexing: {chunks} chunks")
    
    # Query
    result = engine.query("What is the capital of France?")
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    print(f"✓ Query retrieval: {len(result['sources'])} sources found")
    
    # Cleanup
    engine.remove_document("test_doc.txt")
    assert engine.get_document_count() == 0
    print("✓ Document removal")

if __name__ == "__main__":
    print("Running RAG Chatbot Tests\n" + "="*40)
    test_chunking()
    test_load_text_txt()
    test_load_text_json()
    test_rag_engine_basic()
    print("\n✅ All tests passed!")
