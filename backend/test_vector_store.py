
from app.ingestion.file_loader import scan_repository
from app.rag.chunker import create_code_chunks
from app.rag.vector_store import store_code_chunks

# Scan the current CodeSage backend
files = scan_repository(".")

# Convert files into code chunks
chunks = create_code_chunks(files)

print("Total chunks created:", len(chunks))

# Store chunks in ChromaDB
store_code_chunks(chunks)
