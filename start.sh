#!/bin/bash
# DocMind RAG Chatbot - Start Script

echo "🔍 DocMind RAG Chatbot"
echo "======================"

if [ -f ".env" ]; then
  export $(cat .env | grep -v '^#' | xargs)
  echo "✓ Loaded .env"
fi

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠  No LLM API key set. Responses will be retrieval-only."
  echo "   Set ANTHROPIC_API_KEY or OPENAI_API_KEY for full AI answers."
fi

cd backend
echo "🚀 Starting server at http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
