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
    Split PDF pages into chunks while preserving page numbers.
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