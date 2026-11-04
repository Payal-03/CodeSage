
from app.rag.vector_store import search_code

questions = [
    "Where does the application connect to the local AI model?",
    "Which function scans a GitHub repository?",
    "Where is the health check API implemented?"
]

for question in questions:
    print("\nQuestion:", question)

    results = search_code(question, n_results=3)

    for result in results:
        metadata = result["metadata"]

        print(
            f"\nFile: {metadata['file_path']}"
            f"\nName: {metadata['name']}"
            f"\nLines: {metadata['start_line']}-{metadata['end_line']}"
            f"\nDistance: {result['distance']:.4f}"
        )
        print(result["content"])
