
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm import ask_llm
from app.ingestion.github_loader import clone_github_repo
from app.ingestion.file_loader import scan_repository
from app.rag.chunker import create_code_chunks
from app.rag.vector_store import store_code_chunks
from app.rag.qa import answer_codebase_question
from app.analyzers.bug_investigator import investigate_bug

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

class RepositoryQuestionRequest(BaseModel):
    question: str
    repo_id: Optional[str] = None

@app.post("/repository/scan")
def scan_github_repository(request: RepositoryRequest):
    try:
        repo_path = clone_github_repo(request.github_url)
        files = scan_repository(repo_path)
        chunks = create_code_chunks(files)
        repo_id = request.github_url
        store_code_chunks(chunks, repo_id=repo_id)

        return {
            "message": "Repository scanned and indexed successfully",
            "repository_path": repo_path,
            "repo_id": repo_id,
            "total_files": len(files),
            "total_chunks": len(chunks),
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

@app.post("/repository/ask")
def ask_repository_question(request: RepositoryQuestionRequest):
    try:
        qa_result = answer_codebase_question(request.question, repo_id=request.repo_id, return_sources=True)

        return {
            "question": request.question,
            "answer": qa_result["answer"],
            "sources": qa_result.get("sources", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer repository question: {str(e)}"
        )


class BugInvestigationRequest(BaseModel):
    bug_description: str
    error_trace: Optional[str] = None
    repo_id: Optional[str] = None


@app.post("/repository/investigate-bug")
def investigate_repository_bug(request: BugInvestigationRequest):
    try:
        report = investigate_bug(
            bug_description=request.bug_description,
            error_trace=request.error_trace,
            repo_id=request.repo_id
        )
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Bug investigation failed: {str(e)}"
        )
