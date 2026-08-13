import re


def clean_text(text: str) -> str:
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove extra spaces
    text = text.strip()

    return text