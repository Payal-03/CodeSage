import re
from pathlib import Path
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


def _build_contextual_text(chunk: dict) -> str:
    """
    Enriches raw code with file path and symbol metadata for dense vector embedding.
    This significantly improves semantic retrieval relevance.
    """
    file_path = chunk.get("file_path", "")
    name = chunk.get("name", "")
    ctype = chunk.get("type", "")
    content = chunk.get("content", "")

    header = f"# File: {file_path}\n# Symbol: {name} ({ctype})\n"
    return f"{header}{content}"


def tokenize(text: str) -> list[str]:
    """
    Code-aware tokenizer for BM25 keyword search.
    Splits identifiers on snake_case, camelCase, and dots so partial names match.
    """
    tokens = set()
    # Extract words
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    for word in words:
        tokens.add(word)
        # Split snake_case
        if "_" in word:
            for part in word.split("_"):
                if part:
                    tokens.add(part)
        # Split camelCase / numbers
        subwords = re.findall(r"[a-z]+|[0-9]+", word)
        for sub in subwords:
            if len(sub) > 1:
                tokens.add(sub)
    return list(tokens)


def store_code_chunks(chunks: list[dict]):
    """
    Stores code chunks into ChromaDB with dense embeddings and metadata.
    Completely resets existing collection chunks to prevent ghost entries.
    """
    global collection
    if not chunks:
        return

    # Reset collection cleanly to avoid orphaned IDs from previous scans
    try:
        client.delete_collection("codesage_code")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="codesage_code")

    # Build contextual text representations for embeddings
    contextual_texts = [
        _build_contextual_text(chunk)
        for chunk in chunks
    ]
    raw_documents = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(contextual_texts)

    ids = [
        f"chunk_{i}_{chunks[i]['type']}_{Path(chunks[i]['file_path']).stem}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "file_path": str(chunk.get("file_path", "")),
            "name": str(chunk.get("name", "")),
            "type": str(chunk.get("type", "")),
            "start_line": int(chunk.get("start_line", 1)),
            "end_line": int(chunk.get("end_line", 1))
        }
        for chunk in chunks
    ]

    # Batch upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=raw_documents,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} code chunks in ChromaDB.")


def search_code(query: str, n_results: int = 3):
    """
    Performs dense semantic vector search via ChromaDB embeddings.
    """
    count = collection.count()
    if count == 0:
        return []

    effective_n = min(n_results, count)
    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_n
    )

    matches = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            matches.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": float(results["distances"][0][i])
            })

    return matches


def search_bm25(query: str, n_results: int = 3):
    """
    Performs BM25 lexical keyword search over stored repository documents.
    """
    count = collection.count()
    if count == 0:
        return []

    data = collection.get(
        include=["documents", "metadatas"],
        limit=count
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        return []

    # Tokenize each document
    tokenized_documents = [
        tokenize(f"{meta.get('file_path', '')} {meta.get('name', '')}\n{doc}")
        for doc, meta in zip(documents, metadatas)
    ]

    bm25 = BM25Okapi(tokenized_documents)
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    effective_n = min(n_results, len(ranked_indices))

    for index in ranked_indices[:effective_n]:
        # Only include non-zero scoring matches if possible
        if scores[index] > 0 or len(results) == 0:
            results.append({
                "content": documents[index],
                "metadata": metadatas[index],
                "score": float(scores[index])
            })

    return results


def search_hybrid(
    query: str,
    n_results: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60
):
    """
    Performs Hybrid Search combining Dense Vector Search and BM25 Keyword Search
    fused with Reciprocal Rank Fusion (RRF).
    """
    count = collection.count()
    if count == 0:
        return []

    effective_k = min(candidate_k, count)

    semantic_results = search_code(query, n_results=effective_k)
    keyword_results = search_bm25(query, n_results=effective_k)

    combined_results = {}
    search_lists = [
        ("semantic", semantic_results),
        ("keyword", keyword_results)
    ]

    for mode, results in search_lists:
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
                    "rrf_score": 0.0,
                    "matched_by": []
                }

            combined_results[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            combined_results[key]["matched_by"].append(mode)

    ranked_results = sorted(
        combined_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:n_results]
