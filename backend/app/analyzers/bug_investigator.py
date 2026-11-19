import re
from typing import Optional
from app.rag.vector_store import search_hybrid
from app.llm import ask_llm


def investigate_bug(bug_description: str, error_trace: Optional[str] = None) -> dict:
    """
    Performs read-only bug investigation on the indexed codebase:
    1. Extracts search terms from bug description and stack traces.
    2. Retrieves relevant code context via Hybrid Search (Vector + BM25 via RRF).
    3. Analyzes root causes and suggests actionable fixes using the local LLM.
    4. Returns grounded diagnostic evidence without modifying any user files.
    """
    # Combine bug description and error trace for retrieval query
    query_parts = [bug_description.strip()]
    if error_trace and error_trace.strip():
        # Extract potential function/file names or exception names from stack trace
        trace_clean = re.sub(r'File ".*?", line \d+, in ', '', error_trace)
        query_parts.append(trace_clean[:200])

    combined_query = " ".join(query_parts)

    # Retrieve candidate code
    results = search_hybrid(combined_query, n_results=5)

    if not results:
        return {
            "bug_description": bug_description,
            "diagnosis": "No matching code or symbols were found in the indexed repository for this bug description.",
            "affected_files": [],
            "suggested_fix": "Please check if the repository is fully scanned and indexed, or provide additional error trace keywords.",
            "sources": []
        }

    context_parts = []
    sources = []
    affected_files = set()

    for result in results:
        meta = result.get("metadata", {})
        fpath = meta.get("file_path", "")
        start_l = meta.get("start_line", 1)
        end_l = meta.get("end_line", 1)
        sname = meta.get("name", "")
        stype = meta.get("type", "")

        affected_files.add(fpath)
        context_parts.append(
            f"File: {fpath}\n"
            f"Symbol: {sname} ({stype}) [Lines {start_l}-{end_l}]\n"
            f"Code:\n{result.get('content', '')}\n"
            f"--------------------"
        )

        sources.append({
            "file_path": fpath,
            "name": sname,
            "type": stype,
            "start_line": start_l,
            "end_line": end_l,
            "rrf_score": round(result.get("rrf_score", 0.0), 5)
        })

    code_context = "\n".join(context_parts)
    error_section = f"\nERROR TRACE / LOG:\n{error_trace}\n" if error_trace else ""

    prompt = f"""You are CodeSage, a Senior Software Debugging and Code Intelligence Specialist.

A developer is reporting a bug in their codebase. Analyze the provided code snippets and error trace, then deliver a comprehensive diagnostic report.

CODEBASE CONTEXT:
{code_context}

BUG DESCRIPTION:
{bug_description}
{error_section}
INSTRUCTIONS:
1. DIAGNOSIS: Identify the root cause of the bug based strictly on the provided code.
2. AFFECTED FILES & LINES: Explicitly list the exact file paths and line numbers responsible.
3. EVIDENCE: Quote the specific problematic code line(s) and explain why they cause the bug.
4. SUGGESTED FIX: Provide a clean, corrected code replacement snippet. (Do not invent hypothetical libraries).
5. VERIFICATION: Suggest a unit test or step to verify the fix.

Format your response with clear Markdown headings (## Root Cause Diagnosis, ## Affected Code, ## Suggested Fix, ## Verification).

ANALYSIS:
"""

    try:
        diagnosis = ask_llm(prompt)
    except Exception as e:
        diagnosis = f"Error connecting to local LLM for bug analysis: {str(e)}"

    return {
        "bug_description": bug_description,
        "error_trace": error_trace,
        "diagnosis": diagnosis,
        "affected_files": sorted(list(affected_files)),
        "sources": sources
    }

