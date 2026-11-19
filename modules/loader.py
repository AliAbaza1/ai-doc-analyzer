"""Read text out of uploaded documents."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_TYPES = ["txt", "md", "pdf", "docx"]


def read_document(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix in ("txt", "md"):
        for encoding in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if not text:
            raise ValueError("This PDF has no selectable text (it may be a scanned image).")
        return text
    if suffix == "docx":
        from docx import Document

        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs).strip()
    raise ValueError(f"Unsupported file type: .{suffix}")
