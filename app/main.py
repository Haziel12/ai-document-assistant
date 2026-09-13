from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from app.src.llm import ask_gemini
from app.src.retriever import Retriever

RELEVANCE_THRESHOLD = 0.36
app = FastAPI()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Question(BaseModel):
    question: str

class Source(BaseModel):
    source: str
    page: int
    score: float

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source]


retriever = Retriever(
    index_path="data/vector_store/index.faiss",
    metadata_path="data/vector_store/chunks.json",
)


@app.get("/")
def root():
    return {"message": "AI Document Assistant API is running!"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    logger.info("Received question")

    try:
        # Retrieve relevant document chunks
        results = retriever.retrieve(
            question.question,
            top_k=3,
        )

        max_score = max(result["score"] for result in results)

        if max_score < RELEVANCE_THRESHOLD:
            logger.info(
                "Question rejected due to low relevance score: %.3f",
                max_score,
            )

            return {
                "question": question.question,
                "answer": "The information is not available in the provided document.",
                "sources": [],
            }
        # Build sources metadata
        sources = [
            {
                "source": result["source"],
                "page": result["page"],
                "score": result["score"],
            }
            for result in results
        ]

        # Build context from retrieved chunks
        context = "\n\n".join(
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
            for result in results
        )

        logger.info(
            "Retrieved %d relevant chunks",
            len(results),
        )

        # Generate answer using retrieved context
        answer = ask_gemini(
            question.question,
            context,
        )

        logger.info("Gemini response generated successfully")

        return {
            "question": question.question,
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.error(
            "Error while processing question: %s",
            e,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error while processing question: {str(e)}",
        )