def create_chunks(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "unknown",
) -> list[dict]:
    """Create overlapping text chunks while preserving metadata."""

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                    "page": page_number,
                }
            )

            chunk_id += 1
            start += chunk_size - overlap

    return chunks