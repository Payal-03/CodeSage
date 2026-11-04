
import chromadb

from app.rag.embeddings import generate_embeddings


# Create a persistent ChromaDB database
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Create or connect to our code collection
collection = client.get_or_create_collection(
    name="codesage_code"
)


def store_code_chunks(chunks):
    if not chunks:
        return

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "file_path": chunk["file_path"],
            "name": chunk["name"],
            "type": chunk["type"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"]
        }
        for chunk in chunks
    ]
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} code chunks in ChromaDB.")


def search_code(query: str, n_results: int = 3):
    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    matches = []

    for i in range(len(results["ids"][0])):
        matches.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return matches
