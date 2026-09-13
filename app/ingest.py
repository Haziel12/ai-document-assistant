from app.src.pdf import extract_text_from_pdf
from app.src.chunking import create_chunks
from app.src.embeddings import create_embeddings
from app.src.vector_store import VectorStore
from app.src.metadata import save_chunks


PDF_PATH = "data/documents/HazielSanchez_CV_EN.pdf"
INDEX_PATH = "data/vector_store/index.faiss"
METADATA_PATH = "data/vector_store/chunks.json"


def main():
    # 1. Extract text from PDF
    pages = extract_text_from_pdf(PDF_PATH)

    print(f"Pages extracted: {len(pages)}")

    # 2. Create chunks
    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=50,
        source=PDF_PATH,
    )

    print(f"Chunks created: {len(chunks)}")

    # 3. Create embeddings
    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # 4. Create FAISS index with the correct dimension
    dimension = len(embeddings[0])

    print(f"Dimension passed to VectorStore: {dimension}")

    vector_store = VectorStore(dimension)

    print(f"FAISS index dimension: {vector_store.index.d}")

    vector_store.add(embeddings)

    print(f"FAISS index dimension after add: {vector_store.index.d}")

    # 5. Save everything
    vector_store.save(INDEX_PATH)
    save_chunks(chunks, METADATA_PATH)

    print("Vector store created successfully.")


if __name__ == "__main__":
    main()