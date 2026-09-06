from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import logging

from src.llm import ask_gemini


app = FastAPI()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class Question(BaseModel):
    question: str
class Answer(BaseModel):
    question: str
    answer: str

@app.get("/")
def root():
    return {"message": "AI Document Assistant API is running!"}


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    logger.info("Received question")

    try:
        answer = ask_gemini(question.question)
        logger.info("Gemini response generated successfully")
        return {
            "question": question.question,
            "answer": answer,
        }

    except Exception as e:
        logger.error("Error while communicating with Gemini: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error while communicating with Gemini: {str(e)}",
        )