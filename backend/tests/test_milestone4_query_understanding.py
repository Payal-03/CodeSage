"""
Unit and integration tests for CodeSage Milestone 4.1:
- Query intent classification (SINGLE_SYMBOL, ARCHITECTURAL, DEBUGGING)
- Production-first stratified file-diversity retrieval
- Explicit test query preservation
- Guardrails on per-file and total result limits
"""

import unittest
from app.rag.intent import (
    SINGLE_SYMBOL,
    ARCHITECTURAL,
    DEBUGGING,
    classify_query_intent,
    is_explicit_test_query,
)
from app.rag.vector_store import (
    _apply_stratified_file_diversity,
    search_hybrid,
    CANDIDATE_POOL_ARCHITECTURAL,
    CANDIDATE_POOL_DEFAULT,
    MAX_CHUNKS_PER_FILE_ARCHITECTURAL,
)


class TestMilestone4QueryClassification(unittest.TestCase):
    """Test suite for deterministic query classification."""

    def test_specific_function_lookup(self):
        self.assertEqual(classify_query_intent("Where is ask_llm defined?"), SINGLE_SYMBOL)
        self.assertEqual(classify_query_intent("Which function handles the /health endpoint?"), SINGLE_SYMBOL)
        self.assertEqual(classify_query_intent("What does parse_python_code return?"), SINGLE_SYMBOL)

    def test_class_lookup(self):
        self.assertEqual(classify_query_intent("Where is the BM25Okapi class implemented?"), SINGLE_SYMBOL)
        self.assertEqual(classify_query_intent("Which class defines ChatRequest?"), SINGLE_SYMBOL)

    def test_architectural_questions(self):
        self.assertEqual(
            classify_query_intent("Explain the overall system architecture of CodeSage."),
            ARCHITECTURAL
        )
        self.assertEqual(
            classify_query_intent("How does data flow from ingestion to vector storage?"),
            ARCHITECTURAL
        )
        self.assertEqual(
            classify_query_intent("Describe the end-to-end question answering pipeline."),
            ARCHITECTURAL
        )

    def test_cross_file_flow_questions(self):
        self.assertEqual(
            classify_query_intent("How does the frontend communicate with the backend?"),
            ARCHITECTURAL
        )
        self.assertEqual(
            classify_query_intent("Trace the execution flow from FastAPI route into vector store."),
            ARCHITECTURAL
        )
        self.assertEqual(
            classify_query_intent("How do parser and chunker interact across files?"),
            ARCHITECTURAL
        )

    def test_debugging_questions(self):
        self.assertEqual(
            classify_query_intent("TypeError: unsupported operand type(s) for +: 'int' and 'str'"),
            DEBUGGING
        )
        self.assertEqual(
            classify_query_intent("Traceback (most recent call last): File 'main.py', line 33"),
            DEBUGGING
        )
        self.assertEqual(
            classify_query_intent("Why is scan_repository failing with FileNotFoundError?"),
            DEBUGGING
        )
        self.assertEqual(
            classify_query_intent("Server returned 500 error code on /ask endpoint"),
            DEBUGGING
        )

    def test_explicit_test_detection(self):
        self.assertTrue(is_explicit_test_query("Where is the unit test for parser?"))
        self.assertTrue(is_explicit_test_query("Which assertions verify hybrid search?"))
        self.assertTrue(is_explicit_test_query("Run regression tests"))
        self.assertFalse(is_explicit_test_query("Where is the FastAPI application initialized?"))
        self.assertFalse(is_explicit_test_query("How does the frontend communicate with the backend?"))

    def test_ambiguous_and_edge_cases(self):
        # Empty query fallback
        self.assertEqual(classify_query_intent(""), SINGLE_SYMBOL)
        self.assertEqual(classify_query_intent("   "), SINGLE_SYMBOL)
        # Single code identifier
        self.assertEqual(classify_query_intent("search_hybrid"), SINGLE_SYMBOL)
        self.assertEqual(classify_query_intent("/repository/ask"), SINGLE_SYMBOL)
        # General question with architectural term defaults to ARCHITECTURAL
        self.assertEqual(classify_query_intent("How do modules interact?"), ARCHITECTURAL)


class TestMilestone4StratifiedRetrieval(unittest.TestCase):
    """Test suite for production-first stratified multi-file retrieval logic."""

    def test_production_first_selection_over_tests(self):
        """
        For architectural queries, relevant production files should be selected
        before irrelevant test files when both are candidates.
        """
        candidates = [
            # Test file with high score
            {"id": "t1", "metadata": {"file_path": "backend/test_qa.py", "is_test": True}, "rrf_score": 0.040},
            {"id": "t2", "metadata": {"file_path": "backend/test_qa.py", "is_test": True}, "rrf_score": 0.038},
            # Production files with lower raw scores
            {"id": "p1", "metadata": {"file_path": "backend/app/main.py", "is_test": False}, "rrf_score": 0.035},
            {"id": "p2", "metadata": {"file_path": "backend/app/main.py", "is_test": False}, "rrf_score": 0.033},
            {"id": "p3", "metadata": {"file_path": "backend/app/llm.py", "is_test": False}, "rrf_score": 0.030},
            {"id": "p4", "metadata": {"file_path": "backend/app/rag/qa.py", "is_test": False}, "rrf_score": 0.028},
            {"id": "p5", "metadata": {"file_path": "backend/app/rag/vector_store.py", "is_test": False}, "rrf_score": 0.025},
        ]

        diversified = _apply_stratified_file_diversity(
            candidates, target_n=5, max_per_file=2, is_test_query=False
        )
        selected_ids = [c["id"] for c in diversified]

        # Production chunks must be selected; test chunks t1, t2 should NOT displace them
        self.assertEqual(selected_ids, ["p1", "p2", "p3", "p4", "p5"])
        for c in diversified:
            self.assertFalse(c["metadata"]["is_test"])

    def test_explicit_test_query_prioritizes_tests(self):
        """
        When user explicitly asks about tests, test chunks should remain retrievable
        and should not be penalized by production-first filtering.
        """
        candidates = [
            {"id": "t1", "metadata": {"file_path": "backend/test_qa.py", "is_test": True}, "rrf_score": 0.045},
            {"id": "t2", "metadata": {"file_path": "backend/test_qa.py", "is_test": True}, "rrf_score": 0.040},
            {"id": "p1", "metadata": {"file_path": "backend/app/qa.py", "is_test": False}, "rrf_score": 0.030},
            {"id": "t3", "metadata": {"file_path": "backend/test_search.py", "is_test": True}, "rrf_score": 0.028},
        ]

        diversified = _apply_stratified_file_diversity(
            candidates, target_n=3, max_per_file=2, is_test_query=True
        )
        selected_ids = [c["id"] for c in diversified]

        # Should select highest-ranked chunks directly (t1, t2, p1)
        self.assertEqual(selected_ids, ["t1", "t2", "p1"])

    def test_per_file_and_result_limits(self):
        """Verify no more than 2 chunks from same file and max 7 total chunks."""
        chunks = [
            {"id": f"p_a_{i}", "metadata": {"file_path": "a.py", "is_test": False}} for i in range(5)
        ] + [
            {"id": f"p_b_{i}", "metadata": {"file_path": "b.py", "is_test": False}} for i in range(5)
        ] + [
            {"id": f"p_c_{i}", "metadata": {"file_path": "c.py", "is_test": False}} for i in range(5)
        ] + [
            {"id": f"p_d_{i}", "metadata": {"file_path": "d.py", "is_test": False}} for i in range(5)
        ]

        res = _apply_stratified_file_diversity(chunks, target_n=7, max_per_file=2, is_test_query=False)

        self.assertEqual(len(res), 7)
        file_counts = {}
        for c in res:
            fp = c["metadata"]["file_path"]
            file_counts[fp] = file_counts.get(fp, 0) + 1
            self.assertLessEqual(file_counts[fp], 2)

    def test_allow_same_file_and_test_chunks_when_few_production_files(self):
        """
        If fewer distinct production files exist in candidate pool, allow additional
        chunks from same file, and fall back to test chunks rather than leaving slots empty.
        """
        pool = [
            {"id": "p1", "metadata": {"file_path": "a.py", "is_test": False}},
            {"id": "p2", "metadata": {"file_path": "a.py", "is_test": False}},
            {"id": "p3", "metadata": {"file_path": "a.py", "is_test": False}},
            {"id": "t1", "metadata": {"file_path": "test_a.py", "is_test": True}},
            {"id": "t2", "metadata": {"file_path": "test_a.py", "is_test": True}},
        ]

        res = _apply_stratified_file_diversity(pool, target_n=5, max_per_file=2, is_test_query=False)

        # All 5 available chunks should be returned (p1, p2, p3, t1, t2)
        self.assertEqual(len(res), 5)
        self.assertEqual([c["id"] for c in res], ["p1", "p2", "p3", "t1", "t2"])

    def test_empty_and_trivial_inputs(self):
        self.assertEqual(_apply_stratified_file_diversity([], target_n=7), [])
        single = [{"id": "x", "metadata": {"file_path": "x.py", "is_test": False}}]
        self.assertEqual(_apply_stratified_file_diversity(single, target_n=7), single)


if __name__ == "__main__":
    unittest.main()
