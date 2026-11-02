
from pathlib import Path
from urllib.parse import urlparse
from git import Repo
import re

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_STORAGE = BACKEND_DIR / "temp_repos"


def clone_github_repo(repo_url: str) -> str:
    parsed = urlparse(repo_url)

    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Only HTTPS GitHub repository URLs are allowed")

    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Invalid GitHub URL")

    path = parsed.path.strip("/")

    if path.endswith(".git"):
        path = path[:-4]

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        raise ValueError("Invalid GitHub repository URL")

    REPO_STORAGE.mkdir(parents=True, exist_ok=True)

    destination = REPO_STORAGE / path.replace("/", "_")

    if destination.exists():
        raise ValueError("Repository already downloaded. Remove it before cloning again.")

    Repo.clone_from(
        repo_url,
        str(destination),
        depth=1
    )

    return str(destination)
