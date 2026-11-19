import unittest
from app.rag.vector_store import search_hybrid


# Representative evaluation dataset for CodeSage codebase
EVAL_DATASET = [
    {
        "query": "Where is the FastAPI application initialized?",
        "expected_file": "backend/app/main.py",
        "expected_symbol": "app"
    },
    {
        "query": "Which function handles the /health endpoint?",
        "expected_file": "backend/app/main.py",
        "expected_symbol": "health"
    },
    {
        "query": "Where is the Ollama endpoint URL defined?",
        "expected_file": "backend/app/llm.py",
        "expected_symbol": "OLLAMA_URL"
    },
    {
        "query": "Which function executes tree-sitter AST parsing for Python?",
        "expected_file": "backend/app/ingestion/code_parser.py",
        "expected_symbol": "parse_python_code"
    },
    {
        "query": "Where are files scanned and filtered in the repository?",
        "expected_file": "backend/app/ingestion/file_loader.py",
        "expected_symbol": "scan_repository"
    },
    {
        "query": "Where is the hybrid search combining BM25 and vector search implemented?",
        "expected_file": "backend/app/rag/vector_store.py",
        "expected_symbol": "search_hybrid"
    },
    {
        "query": "Which function creates chunks from parsed symbols and files?",
        "expected_file": "backend/app/rag/chunker.py",
        "expected_symbol": "create_code_chunks"
    }
]


class TestRetrievalEvaluation(unittest.TestCase):

    def test_evaluate_retrieval_quality(self):
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        reciprocal_ranks = []

        print("\n" + "=" * 65)
        print("CODESAGE RETRIEVAL EVALUATION REPORT")
        print("=" * 65)

        for case in EVAL_DATASET:
            query = case["query"]
            target_file = case["expected_file"]
            target_sym = case["expected_symbol"]

            results = search_hybrid(query, n_results=5)
            rank_found = None

            for rank, r in enumerate(results, start=1):
                meta = r.get("metadata", {})
                r_file = meta.get("file_path", "")
                r_name = meta.get("name", "")

                # Check if this result matches the expected target file and symbol
                if (target_file in r_file or r_file in target_file) and (target_sym.lower() in r_name.lower() or target_sym in r.get("content", "")):
                    rank_found = rank
                    break

            if rank_found == 1:
                hit_at_1 += 1
            if rank_found and rank_found <= 3:
                hit_at_3 += 1
            if rank_found and rank_found <= 5:
                hit_at_5 += 1

            if rank_found:
                reciprocal_ranks.append(1.0 / rank_found)
                status = f"HIT (Rank {rank_found})"
            else:
                reciprocal_ranks.append(0.0)
                status = "MISS"

            print(f"Query: '{query}'")
            print(f"Target: {target_file} -> {target_sym} | Result: {status}\n")

        total = len(EVAL_DATASET)
        mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
        acc_at_1 = (hit_at_1 / total) * 100
        acc_at_3 = (hit_at_3 / total) * 100
        acc_at_5 = (hit_at_5 / total) * 100

        print("-" * 65)
        print(f"Total Evaluated Queries: {total}")
        print(f"Hit@1 Accuracy: {acc_at_1:.1f}% ({hit_at_1}/{total})")
        print(f"Hit@3 Accuracy: {acc_at_3:.1f}% ({hit_at_3}/{total})")
        print(f"Hit@5 Accuracy: {acc_at_5:.1f}% ({hit_at_5}/{total})")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
        print("=" * 65)

        # Assert reasonable retrieval thresholds (Hit@5 >= 70%)
        self.assertGreaterEqual(hit_at_5 / total, 0.70)


if __name__ == "__main__":
    unittest.main()

