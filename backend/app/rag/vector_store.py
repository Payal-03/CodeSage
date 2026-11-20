import re
import hashlib
from typing import Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb

from app.rag.embeddings import generate_embeddings

DEFAULT_COLLECTION = "codesage_default"
_ACTIVE_REPO: Optional[str] = None

# Create a persistent ChromaDB database
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Connect to default collection for fallback
collection = client.get_or_create_collection(
    name=DEFAULT_COLLECTION
)

# In-memory BM25 index cache: collection_name -> {
#   "bm25": BM25Okapi,
#   "documents": list[str],
#   "metadatas": list[dict],
#   "tokenized_documents": list[list[str]]
# }
_BM25_CACHE: dict[str, dict] = {}


def get_active_repo() -> Optional[str]:
    """Returns the process-local active repository identifier."""
    global _ACTIVE_REPO
    return _ACTIVE_REPO


def set_active_repo(repo_id: Optional[str]):
    """Sets the process-local active repository identifier."""
    global _ACTIVE_REPO
    _ACTIVE_REPO = repo_id


def get_collection_name(repo_id: Optional[str] = None) -> str:
    """
    Derives a stable, sanitized ChromaDB collection name from a repository identifier.
    ChromaDB requirements:
    - 3-63 characters length
    - Starts and ends with an alphanumeric character ([a-zA-Z0-9])
    - Contains only [a-zA-Z0-9_-]
    """
    target = repo_id or _ACTIVE_REPO
    if not target or not target.strip():
        # Backward compatibility check for older codesage_code collection if present
        try:
            client.get_collection(name="codesage_code")
            return "codesage_code"
        except Exception:
            return DEFAULT_COLLECTION

    clean = target.strip().rstrip("/\\")
    if clean.endswith(".git"):
        clean = clean[:-4]

    # Extract owner/repo or directory name
    parts = re.split(r"[/\\:]+", clean)
    if len(parts) >= 2:
        slug = f"{parts[-2]}_{parts[-1]}"
    elif parts:
        slug = parts[-1]
    else:
        slug = "repo"

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
    if not sanitized:
        sanitized = "repo"

    short_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:8]
    base = f"cs_{sanitized}"[:45].rstrip("_")
    collection_name = f"{base}_{short_hash}"

    collection_name = re.sub(r"^[^a-zA-Z0-9]+", "c", collection_name)
    collection_name = re.sub(r"[^a-zA-Z0-9]+$", "0", collection_name)

    return collection_name


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
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    for word in words:
        tokens.add(word)
        if "_" in word:
            for part in word.split("_"):
                if part:
                    tokens.add(part)
        subwords = re.findall(r"[a-z]+|[0-9]+", word)
        for sub in subwords:
            if len(sub) > 1:
                tokens.add(sub)
    return list(tokens)


def _update_bm25_cache(collection_name: str, chunks: list[dict]):
    """Builds and caches tokenized documents, BM25 model, and metadata for a collection."""
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "file_path": str(chunk.get("file_path", "")),
            "name": str(chunk.get("name", "")),
            "type": str(chunk.get("type", "")),
            "start_line": int(chunk.get("start_line", 1)),
            "end_line": int(chunk.get("end_line", 1)),
            "is_test": bool(chunk.get("is_test", False))
        }
        for chunk in chunks
    ]
    tokenized_documents = [
        tokenize(f"{meta.get('file_path', '')} {meta.get('name', '')}\n{doc}")
        for doc, meta in zip(documents, metadatas)
    ]
    bm25 = BM25Okapi(tokenized_documents)
    _BM25_CACHE[collection_name] = {
        "bm25": bm25,
        "documents": documents,
        "metadatas": metadatas,
        "tokenized_documents": tokenized_documents
    }


def _get_or_load_bm25_cache(collection_name: str) -> Optional[dict]:
    """
    Returns the cached BM25 index for collection_name.
    If not in memory (e.g. after server reboot), rebuilds lazily from persistent ChromaDB.
    """
    if collection_name in _BM25_CACHE:
        return _BM25_CACHE[collection_name]

    # Lazy rebuild from persistent ChromaDB
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        # Check backward compatibility
        if collection_name == DEFAULT_COLLECTION:
            try:
                coll = client.get_collection(name="codesage_code")
            except Exception:
                return None
        else:
            return None

    try:
        count = coll.count()
        if count == 0:
            return None
        data = coll.get(include=["documents", "metadatas"], limit=count)
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        if not documents:
            return None

        tokenized_documents = [
            tokenize(f"{meta.get('file_path', '')} {meta.get('name', '')}\n{doc}")
            for doc, meta in zip(documents, metadatas)
        ]
        bm25 = BM25Okapi(tokenized_documents)
        _BM25_CACHE[collection_name] = {
            "bm25": bm25,
            "documents": documents,
            "metadatas": metadatas,
            "tokenized_documents": tokenized_documents
        }
        return _BM25_CACHE[collection_name]
    except Exception:
        return None


def store_code_chunks(chunks: list[dict], repo_id: Optional[str] = None):
    """
    Stores code chunks into a repository-specific ChromaDB collection.
    Only resets the collection for the specified repository, keeping other
    repositories intact. Also builds and updates the in-memory BM25 index.
    """
    global collection
    if not chunks:
        return

    collection_name = get_collection_name(repo_id)
    if repo_id:
        set_active_repo(repo_id)

    # Reset ONLY this repository's collection to avoid orphaned IDs
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    target_collection = client.get_or_create_collection(name=collection_name)
    collection = target_collection

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
            "end_line": int(chunk.get("end_line", 1)),
            "is_test": bool(chunk.get("is_test", False))
        }
        for chunk in chunks
    ]

    # Batch upsert into ChromaDB
    target_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=raw_documents,
        metadatas=metadatas
    )

    # Update in-memory BM25 cache for this collection
    _update_bm25_cache(collection_name, chunks)

    print(f"Stored {len(chunks)} code chunks in ChromaDB collection '{collection_name}'.")


def search_code(query: str, repo_id: Optional[str] = None, n_results: int = 3):
    """
    Performs dense semantic vector search via ChromaDB embeddings for a repository.
    """
    collection_name = get_collection_name(repo_id)
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        if collection_name == DEFAULT_COLLECTION:
            try:
                coll = client.get_collection(name="codesage_code")
            except Exception:
                return []
        else:
            return []

    count = coll.count()
    if count == 0:
        return []

    effective_n = min(n_results, count)
    query_embedding = generate_embeddings([query])[0]

    results = coll.query(
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


def search_bm25(query: str, repo_id: Optional[str] = None, n_results: int = 3):
    """
    Performs BM25 lexical keyword search over stored repository documents using the BM25 cache.
    """
    collection_name = get_collection_name(repo_id)
    cache = _get_or_load_bm25_cache(collection_name)
    if not cache:
        return []

    bm25 = cache["bm25"]
    documents = cache["documents"]
    metadatas = cache["metadatas"]
    tokenized_documents = cache.get("tokenized_documents", [])

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
        doc_tokens = set(tokenized_documents[index]) if index < len(tokenized_documents) else set()
        if scores[index] > 0 or any(t in doc_tokens for t in query_tokens) or len(results) == 0:
            results.append({
                "content": documents[index],
                "metadata": metadatas[index],
                "score": float(scores[index])
            })

    return results


def is_explicit_test_query(query: str) -> bool:
    """Detects if user is explicitly asking about tests, testing, specs, or assertions."""
    pattern = r"\b(test|tests|testing|testcase|testcases|spec|specs|assert|assertion|assertions|unit\s*test|integration\s*test|mock|mocking)\b"
    return bool(re.search(pattern, query.lower()))


def search_hybrid(
    query: str,
    repo_id: Optional[str] = None,
    n_results: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60
):
    """
    Performs Hybrid Search combining Dense Vector Search and BM25 Keyword Search
    fused with Reciprocal Rank Fusion (RRF) and test-aware intent weighting.
    """
    collection_name = get_collection_name(repo_id)
    try:
        coll = client.get_collection(name=collection_name)
        count = coll.count()
    except Exception:
        if collection_name == DEFAULT_COLLECTION:
            try:
                coll = client.get_collection(name="codesage_code")
                count = coll.count()
            except Exception:
                return []
        else:
            return []

    if count == 0:
        return []

    effective_k = min(candidate_k, count)

    semantic_results = search_code(query, repo_id=repo_id, n_results=effective_k)
    keyword_results = search_bm25(query, repo_id=repo_id, n_results=effective_k)

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

    # Test-aware intent routing:
    # If user asks an ordinary implementation question, apply gentle dampening (0.65)
    # to test files so that actual production functions rank highest.
    # If user explicitly asks about tests, boost test chunks (1.25) so they rank top.
    is_test_q = is_explicit_test_query(query)

    for item in combined_results.values():
        meta = item["metadata"]
        fpath = meta.get("file_path", "").lower()
        is_test_chunk = meta.get("is_test", False) or "test" in fpath

        if is_test_q:
            if is_test_chunk:
                item["rrf_score"] *= 1.5
        else:
            if is_test_chunk:
                item["rrf_score"] *= 0.65

    ranked_results = sorted(
        combined_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:n_results]
