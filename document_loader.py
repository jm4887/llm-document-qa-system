"""
document_loader.py — Unified loader for PDF, DOCX, and TXT files.

LangChain's "Document" object is the standard unit of text throughout the pipeline.
Each Document has two fields:
  - page_content: the raw text
  - metadata: a dict (source, page number, author, etc.)

We normalise all file types into this format so the rest of the pipeline
doesn't need to know or care what format the original file was.
"""

from pathlib import Path
from typing import List

from langchain.schema import Document


def load_document(file_path: str) -> List[Document]:
    """
    Dispatch to the correct loader based on file extension.
    Returns a list of LangChain Document objects.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    loaders = {
        ".pdf":  _load_pdf,
        ".docx": _load_docx,
        ".doc":  _load_docx,
        ".txt":  _load_txt,
        ".md":   _load_txt,
    }

    loader_fn = loaders.get(suffix)
    if not loader_fn:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {list(loaders.keys())}"
        )

    return loader_fn(path)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> List[Document]:
    """
    PDFs are loaded page-by-page. Each page becomes its own Document,
    which preserves page number metadata — useful for citations.

    PyPDFLoader handles most PDFs well. For scanned PDFs (images of text),
    you'd need OCR via pytesseract or Amazon Textract.
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    # Enrich metadata
    for i, page in enumerate(pages):
        page.metadata.update({
            "source": path.name,
            "page": i + 1,
            "total_pages": len(pages),
            "file_type": "pdf",
        })

    return pages


def _load_docx(path: Path) -> List[Document]:
    """
    DOCX files are loaded as a single Document (the whole file).
    Docx2txtLoader extracts plain text, stripping formatting.

    Note: tables in DOCX are often poorly extracted as plain text.
    For table-heavy docs, consider python-docx with custom parsing.
    """
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": path.name,
            "file_type": "docx",
        })

    return docs


def _load_txt(path: Path) -> List[Document]:
    """
    Plain text / Markdown files. We try UTF-8 first, fall back to latin-1
    to handle files created on older Windows systems.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return [Document(
        page_content=text,
        metadata={
            "source": path.name,
            "file_type": path.suffix.lstrip("."),
        },
    )]
