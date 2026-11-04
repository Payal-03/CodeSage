import re
from rank_bm25 import BM25Okapi
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
def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(query: str, n_results: int = 3):
    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:n_results]:
        results.append({
            "content": documents[index],
            "metadata": metadatas[index],
            "score": float(scores[index])
        })

    return results

def search_hybrid(
    query: str,
    n_results: int = 3,
    candidate_k: int = 10,
    rrf_k: int = 60
):
    semantic_results = search_code(
        query,
        n_results=candidate_k
    )

    keyword_results = search_bm25(
        query,
        n_results=candidate_k
    )

    combined_results = {}

    search_lists = [
        semantic_results,
        keyword_results
    ]

    for results in search_lists:
        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            key = (
                metadata["file_path"],
                metadata["name"],
                metadata["start_line"]
            )

            if key not in combined_results:
                combined_results[key] = {
                    "content": result["content"],
                    "metadata": metadata,
                    "rrf_score": 0.0
                }

            combined_results[key]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

    ranked_results = sorted(
        combined_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:n_results]