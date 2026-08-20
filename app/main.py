from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import shutil
import traceback
import requests
from collections import defaultdict

from app.ingestion.embedder import generate_embeddings
from app.ingestion.pdf_loader_fitz import load_pdf, load_pdf_pages
from app.ingestion.text_loader import load_text_file
from app.ingestion.docx_loader import load_docx_file
from app.ingestion.web_loader import load_web_page, get_page_title
from app.ingestion.chunker import chunk_text, chunk_pages
from app.services.rag import ask_rag, _load_all_documents
from app.db.upload_vectors import store_vectors
from app.db.qdrant_connection import client, COLLECTION_NAME


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Supported Upload Extensions
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class UrlUploadRequest(BaseModel):
    url: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to the RAG Chatbot API!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# READ PDF TEST
# ============================================================

@app.get("/read-pdf")
def read_pdf():

    file_path = "documents/resume.pdf"

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="resume.pdf not found in documents folder."
        )

    try:

        text = load_pdf(file_path)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the PDF."
            )

        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create chunks from the PDF."
            )

        embeddings = generate_embeddings(chunks)

        return {
            "total_chunks": len(chunks),
            "embedding_dimension": len(embeddings[0]),
            "first_chunk": chunks[0],
            "first_embedding_preview": embeddings[0][:10].tolist()
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print("READ PDF ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to read PDF: {str(e)}"
        )


# ============================================================
# UPLOAD FILE (PDF / TXT / MD / DOCX)
# ============================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported formats: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        )

    # --------------------------------------------------------
    # Create documents folder
    # --------------------------------------------------------

    os.makedirs("documents", exist_ok=True)

    filename = os.path.basename(file.filename)

    file_path = os.path.join(
        "documents",
        filename
    )

    try:

        print("\n" + "=" * 70)
        print("STARTING FILE UPLOAD")
        print("=" * 70)

        print("Filename:", filename)

        # ----------------------------------------------------
        # STEP 1 — Save File
        # ----------------------------------------------------

        print("\n[1/4] Saving file...")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        print("File saved:", file_path)

        # ----------------------------------------------------
        # STEP 2 — Extract content (format-aware)
        # ----------------------------------------------------

        print(f"\n[2/4] Extracting content ({file_ext})...")

        if file_ext == ".pdf":
            pages = load_pdf_pages(file_path)
        elif file_ext in (".txt", ".md"):
            pages = load_text_file(file_path)
        elif file_ext == ".docx":
            pages = load_docx_file(file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"No loader implemented for {file_ext} files."
            )

        print("Pages extracted:", len(pages))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file contains no readable text."
            )

        # ----------------------------------------------------
        # STEP 3 — Create chunks
        # ----------------------------------------------------

        print("\n[3/4] Creating chunks...")

        chunks = chunk_pages(pages)

        print("Chunks created:", len(chunks))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create chunks from the uploaded file."
            )

        # ----------------------------------------------------
        # STEP 4 — Generate embeddings + Qdrant
        # ----------------------------------------------------

        print(
            "\n[4/4] Generating embeddings "
            "and storing vectors..."
        )

        total_vectors = store_vectors(
            chunks,
            filename
        )

        print("Vectors stored:", total_vectors)

        print("\n" + "=" * 70)
        print("UPLOAD SUCCESSFUL")
        print("=" * 70)

        return {
            "filename": filename,
            "message": "File uploaded and stored successfully",
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "vectors_stored": total_vectors
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print("UPLOAD ERROR")
        print("=" * 70)

        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

        print("\nFULL TRACEBACK:")
        traceback.print_exc()

        print("=" * 70 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )

    finally:

        await file.close()


# ============================================================
# UPLOAD FROM WEBSITE URL
# ============================================================

@app.post("/upload-url")
async def upload_url(request: UrlUploadRequest):

    url = request.url.strip()

    if not url:
        raise HTTPException(
            status_code=400,
            detail="No URL provided."
        )

    try:

        print("\n" + "=" * 70)
        print("STARTING URL UPLOAD")
        print("=" * 70)

        print("URL:", url)

        # ----------------------------------------------------
        # STEP 1 — Fetch + Extract Content
        # ----------------------------------------------------

        print("\n[1/3] Fetching and extracting page content...")

        try:
            pages = load_web_page(url)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch URL: {str(e)}"
            )

        print("Pages extracted:", len(pages))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No readable content found at this URL."
            )

        # ----------------------------------------------------
        # STEP 2 — Create chunks
        # ----------------------------------------------------

        print("\n[2/3] Creating chunks...")

        chunks = chunk_pages(pages)

        print("Chunks created:", len(chunks))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create chunks from this page."
            )

        # ----------------------------------------------------
        # STEP 3 — Generate embeddings + Qdrant
        # ----------------------------------------------------

        print(
            "\n[3/3] Generating embeddings "
            "and storing vectors..."
        )

        page_title = get_page_title(url)
        source_name = f"{page_title} ({url})"

        total_vectors = store_vectors(
            chunks,
            source_name
        )

        print("Vectors stored:", total_vectors)

        print("\n" + "=" * 70)
        print("URL UPLOAD SUCCESSFUL")
        print("=" * 70)

        return {
            "source": source_name,
            "url": url,
            "message": "Page fetched and stored successfully",
            "total_chunks": len(chunks),
            "vectors_stored": total_vectors
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print("URL UPLOAD ERROR")
        print("=" * 70)

        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

        print("\nFULL TRACEBACK:")
        traceback.print_exc()

        print("=" * 70 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process URL: {str(e)}"
        )


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get("/documents")
def list_documents():

    all_chunks = _load_all_documents()

    doc_stats = defaultdict(lambda: {"chunks": 0, "pages": set()})

    for chunk in all_chunks:
        source = chunk.get("source", "Unknown")
        doc_stats[source]["chunks"] += 1
        page = chunk.get("page_number")
        if page is not None:
            doc_stats[source]["pages"].add(page)

    documents = [
        {
            "name": name,
            "chunks": stats["chunks"],
            "pages": len(stats["pages"]) if stats["pages"] else None,
        }
        for name, stats in doc_stats.items()
    ]

    documents.sort(key=lambda d: d["name"].lower())

    total_chunks = sum(d["chunks"] for d in documents)

    return {
        "documents": documents,
        "total_documents": len(documents),
        "total_chunks": total_chunks,
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/documents/{document_name}")
def delete_document(document_name: str):

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
    )

    ids_to_delete = [
        point.id
        for point in result[0]
        if point.payload.get("source") == document_name
    ]

    if not ids_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found."
        )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=ids_to_delete,
    )

    return {
        "message": f"Deleted {len(ids_to_delete)} chunks for '{document_name}'.",
        "deleted_chunks": len(ids_to_delete),
    }


# ============================================================
# CHAT / RAG
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    question = request.question.strip()

    try:

        print("\n" + "=" * 70)
        print("NEW CHAT REQUEST")
        print("=" * 70)

        print("Question:", question)

        # ----------------------------------------------------
        # Prepare conversation history
        #
        # IMPORTANT:
        # We do NOT use rewrite_question here.
        #
        # ask_rag() handles the conversation history itself.
        # ----------------------------------------------------

        history = request.history[-8:]

        history_data = []

        for message in history:

            text = message.text.strip()

            if not text:
                continue

            history_data.append({
                "role": message.role,
                "text": text
            })

        print("\nConversation messages:", len(history_data))

        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        result = ask_rag(
            question,
            history=history_data
        )

        # ----------------------------------------------------
        # Validate RAG result
        # ----------------------------------------------------

        if not result:
            raise Exception(
                "RAG pipeline returned no result."
            )

        answer = result.get(
            "answer",
            "I could not find an answer in the uploaded documents."
        )

        sources = result.get(
            "sources",
            []
        )

        print("\nAnswer generated successfully.")
        print("Sources found:", len(sources))

        print("=" * 70 + "\n")

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print("CHAT ERROR")
        print("=" * 70)

        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

        print("\nFULL TRACEBACK:")
        traceback.print_exc()

        print("=" * 70 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )