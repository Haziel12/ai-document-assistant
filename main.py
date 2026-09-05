from fastapi import FastAPI
from pydantic import BaseModel

from src.llm import ask_gemini


app = FastAPI()


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
    answer = ask_gemini(question.question)

    return {
        "question": question.question,
        "answer": answer,
    }