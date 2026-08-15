# RAG Chatbot — Document-Based AI Assistant

A production-quality Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents, ask natural-language questions, and receive grounded answers with supporting sources.

## Features

- Multi-format document ingestion — PDF, TXT, Markdown, Word (.docx), and website URLs
- Hybrid retrieval — vector search (Qdrant) + BM25 keyword search with Reciprocal Rank Fusion (RRF)
- Document isolation — prevents mixing answers across unrelated documents
- No-match / relevance gate — correctly identifies out-of-scope questions instead of hallucinating, using a keyword-aware similarity threshold
- Follow-up question resolution — resolves pronouns and context across conversation turns
- Privacy filtering — redacts sensitive identifiers (CNIC, IBAN, phone numbers, emails) before and after generation
- Source citation — every answer includes the document and page it came from
- Rule-based hallucination check — flags generated answers whose factual claims (names, numbers) aren't traceable to the retrieved context

## Tech Stack

- Backend: Python, FastAPI
- LLM: Gemini API (google-genai) — model: gemini-3.5-flash
- Vector Database: Qdrant
- Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
- Keyword Search: BM25 with Reciprocal Rank Fusion
- Web content extraction: requests + BeautifulSoup
- Word document parsing: python-docx
- Frontend: React (Vite)

**Reasoning:** Qdrant, FastAPI, and Sentence-Transformers were chosen for their strong Python-native support and free-tier availability. Gemini was used over other LLM APIs primarily for its generous context window and existing familiarity; its free-tier rate limits (10 RPM / ~250 RPD on Flash) were a recurring constraint during development and testing — documented under Known Limitations below.

## Architecture

User question
  -> Conversation history
  -> Follow-up question rewriting
  -> Qdrant vector search
  -> Relevance / no-match gate (keyword-aware threshold)
  -> Target-document selection
  -> BM25 + vector hybrid retrieval
  -> Privacy filtering
  -> Gemini grounded generation
  -> Rule-based hallucination check
  -> Answer + sources

## Project Structure

backend/
- app/
  - main.py (FastAPI app, upload & chat endpoints)
  - services/
    - rag.py (Core RAG pipeline)
    - hallucination_check.py (Rule-based groundedness check)
  - db/ (Qdrant connection & vector storage)
  - ingestion/ (File loaders: PDF, TXT/MD, DOCX, web + chunking strategies)
  - retrieval/ (Hybrid vector + BM25 retrieval)
  - llm/ (Gemini generation & privacy filtering)
- evaluation/
  - test_questions.py (Evaluation test set)
  - run_evaluation.py (Retrieval accuracy evaluator)
  - compare_chunking.py (Chunking strategy comparison)
  - compare_embeddings.py (Embedding model comparison)
  - compare_hybrid.py (Hybrid vs vector-only comparison)
  - test_hallucination_check.py (Hallucination check demonstration)
  - citation_accuracy.py (Interactive citation accuracy evaluator)
- documents/ (Uploaded documents, runtime)

## Setup

1. Clone the repository

   git clone <repo-url>
   cd rag-chatbot/backend

2. Create a virtual environment and install dependencies

   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

3. Create a .env file (see .env.example) with:

   GEMINI_API_KEY=your_key_here
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key

4. Run the server

   uvicorn app.main:app --reload

   API available at http://127.0.0.1:8000

## Evaluation Results

### Retrieval accuracy

   python evaluation/run_evaluation.py

Result: 100% (9/9 test cases) — includes positive (document-grounded), negative (no-match), and cross-document test questions.

### Chunking strategy comparison

Compared fixed-size chunking (500 chars, recursive-splitter fallback) against paragraph-based chunking (preserves paragraph boundaries, merges short paragraphs up to 800 chars).

   python evaluation/compare_chunking.py

| Strategy | Chunks stored | Retrieval accuracy |
|---|---|---|
| Fixed-size | 40 | 9/9 (100%) |
| Paragraph-based | 25 | 9/9 (100%) |

Both strategies achieved identical accuracy on this dataset. Paragraph-based chunking produced 37% fewer chunks, which reduces storage and embedding cost while keeping each chunk semantically whole. On a larger or more ambiguous knowledge base, the two strategies would likely diverge more; this result should be read as "no measurable difference at this dataset size," not "chunking strategy doesn't matter."

### Embedding model comparison

Compared all-MiniLM-L6-v2 (384-dim) against all-mpnet-base-v2 (768-dim).

   python evaluation/compare_embeddings.py

| Model | Dimensions | Retrieval accuracy | Embedding time |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 9/9 (100%) | 11.1s |
| all-mpnet-base-v2 | 768 | 8/9 (88.89%) | 11.6s |

The larger model performed worse on this test set. This is most likely because the relevance threshold (`MIN_VECTOR_SIMILARITY_NO_KEYWORD`) was tuned against MiniLM's score distribution, and mpnet's cosine similarity scores follow a different distribution — the comparison is not fully apples-to-apples without re-tuning the threshold per model. On a 9-question test set, a single failure also swings accuracy by ~11%, so this result should not be read as "smaller models are better," but as evidence that embedding model choice interacts with threshold tuning and needs to be evaluated together, not independently.

### Hybrid search vs vector-only

Compared top-1 chunk keyword coverage between vector-only ranking and hybrid (vector + BM25 + RRF) ranking.

   python evaluation/compare_hybrid.py

Result: both approaches achieved identical 45.83% average keyword hit rate on the top-ranked chunk — no measurable difference on this dataset.

This is because document isolation (selecting the target document before ranking) already narrows the candidate pool to a single small document (3–12 chunks) before hybrid ranking runs, leaving little room for BM25 to change the outcome. Hybrid search's benefit is expected to be more visible on larger, higher-ambiguity knowledge bases with many candidate chunks per document.

### Hallucination check

A rule-based groundedness checker (`app/services/hallucination_check.py`) extracts factual claims (proper nouns, numbers) from a generated answer and checks whether they appear in the retrieved context.

   python evaluation/test_hallucination_check.py

Demonstrated on a synthetic example: a grounded answer scored 1.0 groundedness (not flagged); an answer with fabricated details (wrong company name, wrong amount) scored 0.25 and was correctly flagged, with the specific unsupported claims listed (`['250,000', 'Zenith', 'Corp']`).

Limitation: this is a heuristic, not a guarantee. It can miss paraphrased hallucinations (e.g. a fabricated fact reworded to avoid exact-string matching) and can false-flag legitimate inference or summarization. It is intended as a lightweight first-pass signal, not a replacement for manual review.

### Citation accuracy

Manually verified, for each positive test question, whether the cited source actually supports the generated answer.

   python evaluation/citation_accuracy.py

Result: 6/6 (100%) — every cited source supported the corresponding answer's claims.

Known issue: one case ("What is Umair's role?") returned a refusal answer ("I couldn't find that information...") but still displayed a source citation alongside it. This is a minor UI/logic inconsistency — sources should be suppressed when the answer is a refusal — tracked as a known limitation, not yet fixed.

## Known Limitations

- Gemini's free-tier rate limits (10 RPM / ~250 RPD on Flash) were hit repeatedly during development and testing, occasionally blocking answer generation mid-session. The retrieval pipeline degrades gracefully in this case (returns "AI service temporarily unavailable" rather than crashing), but this is a real constraint for demo reliability.
- Evaluation test set is small (9 questions across 3 documents). Reported 100% accuracy figures should be read in that context — they demonstrate correctness on the current dataset, not guaranteed generalization to a larger or messier knowledge base.
- Chat history is currently held in frontend state, not persisted to a database — history is lost on page refresh.
- Source citations are shown even when a refusal answer is returned in some edge cases (see Citation Accuracy above).
- Embedding-model comparison threshold was tuned for the default model; a fair multi-model comparison would require re-tuning the relevance threshold per model.

## Status

Backend core functionality complete and evaluated — no-match detection, document isolation, hybrid retrieval, follow-up resolution, hallucination checking, and citation accuracy all tested with documented results. Deployment and final frontend polish in progress.