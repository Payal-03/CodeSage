from app.rag.vector_store import search_hybrid

questions = [
    "Where is the scan_repository function?",
    "Where does the application connect to Ollama?",
    "Where is the health check API implemented?"
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    results = search_hybrid(question, n_results=3)

    for result in results:
        metadata = result["metadata"]

        print(
            f"\nFile: {metadata['file_path']}"
            f"\nName: {metadata['name']}"
            f"\nLines: {metadata['start_line']}-{metadata['end_line']}"
            f"\nRRF Score: {result['rrf_score']:.5f}"
        )