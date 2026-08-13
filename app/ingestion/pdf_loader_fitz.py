import fitz


def load_pdf(file_path):
    """
    Load a PDF and return all text as one string.
    """

    doc = fitz.open(file_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


def load_pdf_pages(file_path):
    """
    Load a PDF while preserving page numbers.
    """

    doc = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text()

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })

    doc.close()

    return pages