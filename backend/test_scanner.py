
from app.ingestion.file_loader import scan_repository

# Scan the CodeSage backend itself
files = scan_repository(".")

print(f"\nTotal files scanned: {len(files)}")

for file in files:
    print(
        f"{file['file_path']} | "
        f"{file['total_lines']} lines"
    )
