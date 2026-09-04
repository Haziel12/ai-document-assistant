import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


class AssistantResponse(BaseModel):
    answer: str
    topic: str
    difficulty: str


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.6-flash",
    config={
        "system_instruction": """
        You are an AI assistant that explains technical concepts.

        Always identify:
        - the answer to the user's question
        - the main topic
        - the difficulty level: beginner, intermediate, or advanced
        """
    }
)

print("AI Document Assistant")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = chat.send_message(
        question,
        config={
            "response_mime_type": "application/json",
            "response_schema": AssistantResponse,
        },
    )

    result = AssistantResponse.model_validate_json(response.text)

    print("\nGemini:")
    print("Answer:", result.answer)
    print("Topic:", result.topic)
    print("Difficulty:", result.difficulty)
    print()