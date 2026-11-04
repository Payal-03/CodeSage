from app.rag.vector_store import search_hybrid
from app.llm import ask_llm


def answer_codebase_question(question: str):

    results = search_hybrid(
        question,
        n_results=5
    )

    if not results:
        return "I couldn't find relevant code in the repository."

    context = ""

    for result in results:
        metadata = result["metadata"]

        context += f"""
File: {metadata['file_path']}
Lines: {metadata['start_line']}-{metadata['end_line']}
Code:
{result['content']}

--------------------
"""

    prompt = f"""
You are CodeSage, an AI codebase analysis assistant.

Answer the user's question using ONLY the provided code context.

Instructions:
1. Explain the relevant code clearly.
2. Mention file names and line numbers.
3. If the context does not contain enough information, say so.
4. Do not invent functions, variables, or implementations.

CODE CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    answer = ask_llm(prompt)

    return answer