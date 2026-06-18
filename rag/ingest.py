"""Turn files (PDF / image / text) into clean text, then into chunks."""
import io
from pathlib import Path

from pypdf import PdfReader
from PIL import Image

from .llm import GroqClient
from .splitter import split_text
from . import config

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".log", ".json", ".html"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def load_pdf(path):
    """Extract text from every page of a PDF, tagged with page numbers."""
    reader = PdfReader(str(path))
    parts = []
    for n, page in enumerate(reader.pages, 1):
        txt = (page.extract_text() or "").strip()
        if txt:
            parts.append(f"[Page {n}]\n{txt}")
    return "\n\n".join(parts)


def load_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_image(path, llm=None):
    """OCR + describe an image via Groq's vision model. Returns extracted text."""
    llm = llm or GroqClient()
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))  # downscale to keep the payload small
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return llm.describe_image(buf.getvalue(), mime_type="image/jpeg")


def extract_text(path, llm=None):
    """Dispatch on file extension and return the document's text content."""
    ext = Path(path).suffix.lower()
    if ext in PDF_EXTS:
        return load_pdf(path)
    if ext in IMAGE_EXTS:
        return load_image(path, llm=llm)
    if ext in TEXT_EXTS:
        return load_text(path)
    # Last resort: try to read it as UTF-8 text.
    return load_text(path)


def file_to_chunks(path, llm=None):
    """Return ``(full_text, chunks)`` for a single file."""
    text = extract_text(path, llm=llm)
    chunks = split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return text, chunks
