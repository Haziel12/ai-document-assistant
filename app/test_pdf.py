from src.pdf import extract_text_from_pdf
from src.chunking import create_chunks
from src.embeddings import create_embeddings

pdf_path = "data/documents/HazielSanchez_CV_EN.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=50,
    source="cv.pdf",
)

texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)

print(f"Total chunks: {len(chunks)}")
print(f"Total embeddings: {len(embeddings)}")

print("\nFirst embedding:")
print(embeddings[0])

print("\nEmbedding dimensions:")
print(len(embeddings[0]))