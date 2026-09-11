from google import genai

from app.config import GEMINI_API_KEY


client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(question: str, context: str) -> str:
    """Generate an answer using the retrieved document context."""

    prompt = f"""
You are an AI assistant that answers questions using the provided document context.

Use the context below to answer the user's question.

If the answer cannot be found in the context, say that the information is not available in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text