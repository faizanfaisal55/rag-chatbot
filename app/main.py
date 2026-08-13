from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import shutil
import traceback

from app.ingestion.embedder import generate_embeddings
from app.ingestion.pdf_loader_fitz import load_pdf, load_pdf_pages
from app.ingestion.text_loader import load_text_file
from app.ingestion.chunker import chunk_text, chunk_pages
from app.services.rag import ask_rag
from app.db.upload_vectors import store_vectors


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

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


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
# UPLOAD FILE (PDF / TXT / MD)
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