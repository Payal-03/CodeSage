import os
from pathlib import Path
from app.ingestion.code_parser import parse_python_code

SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".tsx",
    ".jsx", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".php", ".rb", ".md",
    ".json", ".yaml", ".yml", ".toml", ".txt"
}

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv",
    "__pycache__", "dist", "build",
    ".idea", ".next", "coverage",
    "temp_repos", "chroma_db", "parsers", ".pytest_cache"
}

IGNORED_FILES = {
    ".env", ".env.local", ".env.production"
}

MAX_FILE_SIZE = 500 * 1024  # 500 KB


def scan_repository(repo_path: str):
    """
    Scans a repository directory, filters out ignored paths and unsupported extensions,
    and returns a structured list of file metadata and parsed symbols.
    """
    repository = Path(repo_path).resolve()
    results = []

    if not repository.exists() or not repository.is_dir():
        raise ValueError(f"Invalid repository directory: {repo_path}")

    # Use os.walk with directory pruning for fast traversal (avoids traversing huge venvs)
    for root, dirs, files in os.walk(repository):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for file_name in files:
            if file_name in IGNORED_FILES or file_name.startswith("."):
                continue

            file_path = Path(root) / file_name
            ext = file_path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                if file_path.stat().st_size > MAX_FILE_SIZE:
                    continue

                content = file_path.read_text(encoding="utf-8")
                relative_path = file_path.relative_to(repository).as_posix()

                file_info = {
                    "file_path": relative_path,
                    "extension": ext,
                    "content": content,
                    "total_lines": len(content.splitlines()),
                    "language": "python" if ext == ".py" else ext.lstrip("."),
                    "symbols": []
                }

                if ext == ".py":
                    file_info["symbols"] = parse_python_code(content)

                results.append(file_info)

            except (UnicodeDecodeError, OSError):
                continue

    return results
