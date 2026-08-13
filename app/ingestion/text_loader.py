def load_text_file(file_path: str):
    """
    Load a plain text or markdown file.

    Returns a list of "pages" (same shape as PDF loader)
    so it plugs directly into the existing chunk_pages()
    pipeline. Text files don't have real pages, so we treat
    the whole file as a single page.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text.strip():
        return []

    return [{
        "page_number": 1,
        "text": text
    }]