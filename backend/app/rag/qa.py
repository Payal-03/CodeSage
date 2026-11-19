from app.rag.vector_store import search_hybrid
from app.llm import ask_llm


def generate_qa_prompt(context: str, question: str) -> str:
    """Constructs a strict grounding prompt for CodeSage."""
    return f"""You are CodeSage, an expert AI codebase intelligence and debugging assistant.

Answer the user's question using ONLY the provided CODE CONTEXT below.

STRICT INSTRUCTIONS:
1. Ground every statement in the code snippets provided.
2. Mention the exact file names and line numbers when referencing functions, classes, or variables.
3. If the provided context does not contain sufficient information to answer the question accurately, explicitly state:
   "The available codebase context does not contain enough information to answer this question."
4. DO NOT hallucinate or fabricate file names, functions, classes, or code that is not in the context.
5. Provide clear, concise, and structured explanations suitable for technical discussions and code walkthroughs.

CODE CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


def answer_codebase_question(question: str, return_sources: bool = False):
    """
    Retrieves relevant code chunks using Hybrid Search (Vector + BM25 via RRF),
    builds grounded context, and answers the question with LLM.
    """
    results = search_hybrid(
        question,
        n_results=5
    )

    if not results:
        no_ctx_msg = "I couldn't find relevant code in the indexed repository."
        if return_sources:
            return {"answer": no_ctx_msg, "sources": []}
        return no_ctx_msg

    context_parts = []
    sources = []

    for result in results:
        metadata = result["metadata"]
        file_path = metadata.get("file_path", "")
        start_line = metadata.get("start_line", 1)
        end_line = metadata.get("end_line", 1)
        symbol_name = metadata.get("name", "")
        symbol_type = metadata.get("type", "")

        context_parts.append(
            f"File: {file_path}\n"
            f"Symbol: {symbol_name} ({symbol_type})\n"
            f"Lines: {start_line}-{end_line}\n"
            f"Code:\n{result['content']}\n"
            f"--------------------"
        )

        sources.append({
            "file_path": file_path,
            "name": symbol_name,
            "type": symbol_type,
            "start_line": start_line,
            "end_line": end_line,
            "rrf_score": round(result.get("rrf_score", 0.0), 5),
            "matched_by": result.get("matched_by", [])
        })

    context = "\n".join(context_parts)
    prompt = generate_qa_prompt(context, question)

    try:
        answer = ask_llm(prompt)
    except Exception as e:
        answer = f"Error communicating with local LLM (Ollama): {str(e)}"

    if return_sources:
        return {
            "answer": answer,
            "sources": sources
        }

    return answer
