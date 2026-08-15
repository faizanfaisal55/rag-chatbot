import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ==========================================================
# Basic SSRF Protection
# ==========================================================
#
# Block requests to local/internal addresses so the upload
# endpoint can't be used to probe internal network services.
# This is a basic safeguard, not exhaustive security.

BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
}


def _is_safe_url(url: str) -> bool:

    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return False

    if hostname in BLOCKED_HOSTS:
        return False

    # Block private/internal IP ranges (basic check)
    if (
        hostname.startswith("192.168.")
        or hostname.startswith("10.")
        or hostname.startswith("172.16.")
        or hostname.startswith("169.254.")
    ):
        return False

    return True


# ==========================================================
# Fetch + Extract Clean Text
# ==========================================================

def load_web_page(url: str):
    """
    Fetch a web page and extract clean readable text.

    Returns a list of "pages" (same shape as PDF loader)
    so it plugs directly into the existing chunk_pages()
    pipeline. A web page is treated as a single page.
    """

    if not _is_safe_url(url):
        raise ValueError(
            "This URL is not allowed for security reasons."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAGChatbot/1.0)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "text/html" not in content_type:
        raise ValueError(
            "URL does not point to an HTML page."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up excessive blank lines
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    clean_text = "\n".join(lines)

    if not clean_text.strip():
        return []

    return [{
        "page_number": 1,
        "text": clean_text
    }]


def get_page_title(url: str) -> str:
    """
    Best-effort page title extraction, used as the
    display "source" name. Falls back to the URL itself.
    """

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; RAGChatbot/1.0)"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title and soup.title.string:
            return soup.title.string.strip()[:100]

    except Exception:
        pass

    return url