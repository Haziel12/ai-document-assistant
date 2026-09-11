import json
from pathlib import Path


def save_chunks(chunks: list[dict], path: str) -> None:
    """Save document chunks and metadata to a JSON file."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_chunks(path: str) -> list[dict]:
    """Load document chunks and metadata from a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)