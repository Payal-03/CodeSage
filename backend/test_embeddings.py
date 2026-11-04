
from app.rag.embeddings import generate_embeddings

texts = [
    "def add(a, b): return a + b",
    "This function performs addition of two numbers.",
    "The application connects to a local AI model."
]

embeddings = generate_embeddings(texts)

print("Total embeddings:", len(embeddings))
print("Dimensions:", len(embeddings[0]))
print("First embedding preview:", embeddings[0][:5])
