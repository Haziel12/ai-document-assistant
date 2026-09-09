from src.pdf import extract_text_from_pdf
from src.chunking import create_chunks
from src.embeddings import create_embeddings
from src.search import semantic_search

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


query = "What experience does Haziel have with computer vision?"

query_embedding = create_embeddings([query])[0]


results = semantic_search(
    query_embedding,
    chunks,
    chunk_embeddings,
    top_k=3,
)


print("\nSEARCH RESULTS\n")

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")
    print(f"Text: {result['text']}")
    print("-" * 60)