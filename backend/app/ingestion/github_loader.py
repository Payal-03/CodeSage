import shutil
from pathlib import Path
from urllib.parse import urlparse
from git import Repo, GitCommandError
import re

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_STORAGE = BACKEND_DIR / "temp_repos"


def clone_github_repo(repo_url: str, force_reclone: bool = False) -> str:
    """
    Clones or refreshes a public HTTPS GitHub repository.
    Returns the absolute path to the local repository.
    """
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
        if force_reclone:
            shutil.rmtree(destination, ignore_errors=True)
            Repo.clone_from(repo_url, str(destination), depth=1)
        else:
            # If repo already exists, attempt to pull latest changes or reuse
            try:
                repo = Repo(str(destination))
                repo.remotes.origin.pull()
            except Exception:
                # If pull fails (e.g. offline or detached), reuse existing checkout
                pass
        return str(destination)

    Repo.clone_from(
        repo_url,
        str(destination),
        depth=1
    )

    return str(destination)
