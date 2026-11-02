
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm import ask_llm

app = FastAPI(
    title="CodeSage",
    description="AI-powered Codebase Intelligence and Debugging Agent",
    version="0.1.0"
)

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "Welcome to CodeSage!",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(request: ChatRequest):
    try:
        answer = ask_llm(request.question)
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
