from src.pdf import extract_text_from_pdf
from src.chunking import create_chunks

pdf_path = "data/documents/HazielSanchez_CV_EN.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(text)

print(f"Total characters: {len(text)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)