from src.pdf import extract_text_from_pdf
from src.chunking import create_chunks
from src.embeddings import create_embeddings
from src.search import semantic_search
from src.vector_store import VectorStore

pdf_path = "data/documents/HazielSanchez_CV_EN.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=50,
    source="cv.pdf",
)

texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = create_embeddings(texts)

vector_store = VectorStore(
    dimension=len(chunk_embeddings[0])
)

vector_store.add(chunk_embeddings)

query = "What skills Haziel has with OpenCV?"

query_embedding = create_embeddings([query])[0]

scores, indices = vector_store.search(
    query_embedding,
    top_k=3,
)

print("\nFAISS SEARCH RESULTS\n")

for score, index in zip(scores, indices):
    chunk = chunks[index]

    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text']}")
    print("-" * 60)