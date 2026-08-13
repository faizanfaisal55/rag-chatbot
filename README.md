# RAG Chatbot — Document-Based AI Assistant

A production-quality Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents, ask natural-language questions, and receive grounded answers with supporting sources.

## Features

- Document upload (PDF, TXT, Markdown)
- Hybrid retrieval — vector search (Qdrant) + BM25 keyword search with Reciprocal Rank Fusion (RRF)
- Document isolation — prevents mixing answers across unrelated documents
- No-match / relevance gate — correctly identifies out-of-scope questions instead of hallucinating
- Follow-up question resolution — resolves pronouns and context across conversation turns
- Privacy filtering — redacts sensitive identifiers (CNIC, IBAN, phone numbers, emails) before and after generation
- Source citation — every answer includes the document and page it came from

## Tech Stack

- Backend: Python, FastAPI
- LLM: Gemini API (google-genai)
- Vector Database: Qdrant
- Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
- Keyword Search: BM25
- Frontend: React (Vite)

## Architecture

User question
  -> Conversation history
  -> Follow-up question rewriting
  -> Qdrant vector search
  -> Relevance / no-match gate
  -> Target-document selection
  -> BM25 + vector hybrid retrieval
  -> Privacy filtering
  -> Gemini grounded generation
  -> Answer + sources

## Project Structure

backend/
- app/
  - main.py (FastAPI app, upload & chat endpoints)
  - services/rag.py (Core RAG pipeline)
  - db/ (Qdrant connection & vector storage)
  - ingestion/ (File loaders: PDF, TXT/MD + chunking)
  - retrieval/ (Hybrid vector + BM25 retrieval)
  - llm/ (Gemini generation & privacy filtering)
- evaluation/
  - test_questions.py (Evaluation test set)
  - run_evaluation.py (Retrieval accuracy evaluator)
- documents/ (Uploaded documents, runtime)

## Setup

1. Clone the repository

   git clone <repo-url>
   cd rag-chatbot/backend

2. Create a virtual environment and install dependencies

   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

3. Create a .env file with:

   GEMINI_API_KEY=your_key_here
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key

4. Run the server

   uvicorn app.main:app --reload

   API available at http://127.0.0.1:8000

## Evaluation

Run the retrieval accuracy evaluation:

   python evaluation/run_evaluation.py

Current retrieval accuracy: 100% (9/9 test cases)

## Status

Backend core functionality complete — no-match detection, document isolation, hybrid retrieval, and follow-up resolution all tested and verified. Frontend UI polish and deployment in progress.