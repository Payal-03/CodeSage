
from pathlib import Path
from app.ingestion.code_parser import parse_python_code

SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".tsx",
    ".jsx", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".php", ".rb", ".md"
}

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv",
    "__pycache__", "dist", "build",
    ".idea", ".next", "coverage",
    "temp_repos", "chroma_db"
}

IGNORED_FILES = {
    ".env", ".env.local", ".env.production"
}

MAX_FILE_SIZE = 500 * 1024  # 500 KB


def scan_repository(repo_path: str):
    repository = Path(repo_path)
    results = []

    if not repository.exists() or not repository.is_dir():
        raise ValueError("Invalid repository directory")

    for file_path in repository.rglob("*"):

        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(repository)

        if any(part in IGNORED_DIRS for part in relative_path.parts):
            continue

        if file_path.name in IGNORED_FILES:
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            content = file_path.read_text(encoding="utf-8")

            file_info = {
                "file_path": str(relative_path),
                "extension": file_path.suffix.lower(),
                "content": content,
                "total_lines": len(content.splitlines()),
                "language": "python" if file_path.suffix.lower() == ".py" else "other",
                "symbols": []
            }

            if file_path.suffix.lower() == ".py":
                file_info["symbols"] = parse_python_code(content)
                
            results.append(file_info)

        except (UnicodeDecodeError, OSError):
            continue

    return results
