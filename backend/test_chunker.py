from app.ingestion.file_loader import scan_repository
from app.rag.chunker import create_code_chunks

# Test chunker on local codebase
files = scan_repository(".")
chunks = create_code_chunks(files)

print("=" * 60)
print(f"Total files scanned: {len(files)}")
print(f"Total chunks created: {len(chunks)}")
print("=" * 60)

# Categorize chunks by type
chunk_types = {}
for chunk in chunks:
    ctype = chunk["type"]
    chunk_types[ctype] = chunk_types.get(ctype, 0) + 1

print("\nChunk Breakdown by Type:")
for ctype, count in sorted(chunk_types.items()):
    print(f" - {ctype}: {count}")

print("\nSample Chunks Preview:")
for chunk in chunks[:6]:
    print(f"\n[{chunk['type'].upper()}] {chunk['name']}")
    print(f"File: {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})")
    first_line = chunk['content'].strip().splitlines()[0] if chunk['content'].strip() else ""
    print(f"Preview: {first_line[:80]}...")
