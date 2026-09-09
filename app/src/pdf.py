from pathlib import Path

from pypdf import PdfReader


from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from each PDF page with metadata."""

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages