
from app.ingestion.github_loader import clone_github_repo
from app.ingestion.file_loader import scan_repository
from app.rag.chunker import create_code_chunks

repo_url = "https://github.com/Payal-03/CodeSage"

repo_path = clone_github_repo(repo_url)
files = scan_repository(repo_path)

chunks = create_code_chunks(files)

print("Total chunks:", len(chunks))

for chunk in chunks:
    print(f"\n{chunk['type']}: {chunk['name']}")
    print(f"File: {chunk['file_path']}")
    print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
    print("Code:")
    print(chunk["content"])
