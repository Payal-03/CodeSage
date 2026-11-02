
from app.ingestion.github_loader import clone_github_repo
from app.ingestion.file_loader import scan_repository

url = "https://github.com/Payal-03/CodeSage.git"

repo_path = clone_github_repo(url)

print("Repository cloned successfully!")
print("Location:", repo_path)

files = scan_repository(repo_path)

print(f"\nTotal files discovered: {len(files)}")

for file in files:
    print(file["file_path"])
