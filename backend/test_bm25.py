from app.rag.vector_store import search_bm25

questions = [
    "Where is the scan_repository function?",
    "Where does the application connect to Ollama?",
    "Where is the health check API implemented?"
]

for question in questions:
    print("\nQuestion:", question)

    results = search_bm25(question, n_results=3)

    for result in results:
        metadata = result["metadata"]

        print(
            f"\nFile: {metadata['file_path']}"
            f"\nName: {metadata['name']}"
            f"\nScore: {result['score']:.4f}"
        )