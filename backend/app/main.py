
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm import ask_llm
from app.ingestion.github_loader import clone_github_repo
from app.ingestion.file_loader import scan_repository

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

class RepositoryRequest(BaseModel):
    github_url: str


@app.post("/repository/scan")
def scan_github_repository(request: RepositoryRequest):
    try:
        repo_path = clone_github_repo(request.github_url)
        files = scan_repository(repo_path)

        return {
            "message": "Repository scanned successfully",
            "repository_path": repo_path,
            "total_files": len(files),
            "files": [
                {
                    "file_path": file["file_path"],
                    "extension": file["extension"],
                    "total_lines": file["total_lines"]
                }
                for file in files
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Repository scanning failed"
        )
