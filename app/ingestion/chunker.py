from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def chunk_text(text):
    """
    Existing chunking function.
    Keeps compatibility with the current RAG pipeline.
    """

    splitter = get_splitter()

    chunks = splitter.split_text(text)

    return chunks


def chunk_pages(pages):
    """
    STRATEGY 1: Fixed-size chunking (current production strategy).

    Splits text into fixed ~500-character windows with 100-char
    overlap, falling back through separators (paragraph -> line ->
    sentence -> word -> character) to avoid cutting mid-word when
    possible. Does not respect paragraph boundaries as a hard rule —
    a single paragraph can be split across multiple chunks.
    """

    splitter = get_splitter()

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page_number": page["page_number"]
            })

    return chunks


# ==========================================================
# STRATEGY 2: Paragraph-based (semantic) chunking
# ==========================================================

def chunk_pages_paragraph_based(
    pages,
    min_chunk_chars=200,
    max_chunk_chars=800,
):
    """
    STRATEGY 2: Paragraph-based (semantic) chunking.

    Splits text on paragraph boundaries (blank lines) and keeps
    each paragraph as a unit — never splitting a paragraph mid-
    sentence. Consecutive short paragraphs are merged together up
    to max_chunk_chars, so a chunk always represents one or more
    complete paragraphs rather than an arbitrary character window.

    A single paragraph that exceeds max_chunk_chars on its own
    (rare, e.g. a long unbroken block of text) falls back to the
    fixed-size splitter for that paragraph only.
    """

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=50,
        separators=[". ", " ", ""]
    )

    chunks = []

    for page in pages:

        paragraphs = [
            paragraph.strip()
            for paragraph in page["text"].split("\n\n")
            if paragraph.strip()
        ]

        # Some inputs (plain .txt) may not have blank-line breaks
        # at all — fall back to single newlines in that case.
        if len(paragraphs) <= 1:
            paragraphs = [
                paragraph.strip()
                for paragraph in page["text"].split("\n")
                if paragraph.strip()
            ]

        buffer = ""

        for paragraph in paragraphs:

            # Oversized single paragraph — split independently,
            # flush any pending buffer first.
            if len(paragraph) > max_chunk_chars:

                if buffer:
                    chunks.append({
                        "text": buffer,
                        "page_number": page["page_number"]
                    })
                    buffer = ""

                for sub_chunk in fallback_splitter.split_text(paragraph):
                    chunks.append({
                        "text": sub_chunk,
                        "page_number": page["page_number"]
                    })

                continue

            candidate = (
                f"{buffer}\n\n{paragraph}" if buffer else paragraph
            )

            if len(candidate) <= max_chunk_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append({
                        "text": buffer,
                        "page_number": page["page_number"]
                    })
                buffer = paragraph

        if buffer:
            chunks.append({
                "text": buffer,
                "page_number": page["page_number"]
            })

    return chunks