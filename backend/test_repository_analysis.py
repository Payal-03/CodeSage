
from app.ingestion.github_loader import clone_github_repo
from app.ingestion.file_loader import scan_repository

repo_url = "https://github.com/Payal-03/CodeSage"

repo_path = clone_github_repo(repo_url)

files = scan_repository(repo_path)

for file in files:
    print(f"\nFile: {file['file_path']}")
    print(f"Lines: {file['total_lines']}")

    for symbol in file["symbols"]:
        print(
            f"  {symbol['type']}: {symbol['name']} "
            f"(Lines {symbol['start_line']}-{symbol['end_line']})"
        )
