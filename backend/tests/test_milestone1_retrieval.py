import unittest
import time
from app.rag.vector_store import (
    store_code_chunks,
    search_code,
    search_bm25,
    search_hybrid,
    get_collection_name,
    _BM25_CACHE,
    is_explicit_test_query
)


class TestMilestone1RetrievalAndStorage(unittest.TestCase):

    def setUp(self):
        self.repo_a = "https://github.com/test-org/repo-alpha"
        self.repo_b = "https://github.com/test-org/repo-beta"

        self.chunks_a = [
            {
                "file_path": "auth/service.py",
                "name": "login_user",
                "type": "function",
                "start_line": 10,
                "end_line": 25,
                "content": "def login_user(username, password):\n    # Alpha authentication service\n    return token",
                "is_test": False
            },
            {
                "file_path": "tests/test_auth.py",
                "name": "test_login_user",
                "type": "function",
                "start_line": 5,
                "end_line": 15,
                "content": "def test_login_user():\n    assert login_user('admin', 'secret')",
                "is_test": True
            }
        ]

        self.chunks_b = [
            {
                "file_path": "billing/payment.py",
                "name": "process_payment",
                "type": "function",
                "start_line": 1,
                "end_line": 20,
                "content": "def process_payment(amount, currency):\n    # Beta payment processor\n    return True",
                "is_test": False
            }
        ]

    def test_collection_isolation(self):
        """Scanning Repo B must not delete or overwrite Repo A's data."""
        store_code_chunks(self.chunks_a, repo_id=self.repo_a)
        store_code_chunks(self.chunks_b, repo_id=self.repo_b)

        # Query Repo A
        results_a = search_hybrid("login authentication service", repo_id=self.repo_a, n_results=5)
        self.assertGreater(len(results_a), 0)
        self.assertEqual(results_a[0]["metadata"]["file_path"], "auth/service.py")

        # Query Repo B
        results_b = search_hybrid("process payment currency", repo_id=self.repo_b, n_results=5)
        self.assertGreater(len(results_b), 0)
        self.assertEqual(results_b[0]["metadata"]["file_path"], "billing/payment.py")

        # Confirm Repo A does not leak into Repo B searches
        results_b_for_a = search_hybrid("login authentication", repo_id=self.repo_b, n_results=5)
        for r in results_b_for_a:
            self.assertNotEqual(r["metadata"]["file_path"], "auth/service.py")

    def test_bm25_cache_reuse_and_invalidation(self):
        """BM25 index must be cached in memory and selectively invalidated upon re-index."""
        store_code_chunks(self.chunks_a, repo_id=self.repo_a)
        coll_a = get_collection_name(self.repo_a)

        self.assertIn(coll_a, _BM25_CACHE)
        cached_instance = _BM25_CACHE[coll_a]["bm25"]

        # Consecutive search reuses cached instance
        search_bm25("login authentication", repo_id=self.repo_a)
        self.assertIs(_BM25_CACHE[coll_a]["bm25"], cached_instance)

        # Re-index invalidates cache with new object
        updated_chunks_a = self.chunks_a + [
            {
                "file_path": "auth/mfa.py",
                "name": "verify_otp",
                "type": "function",
                "start_line": 1,
                "end_line": 10,
                "content": "def verify_otp(code):\n    return True",
                "is_test": False
            }
        ]
        store_code_chunks(updated_chunks_a, repo_id=self.repo_a)
        self.assertIsNot(_BM25_CACHE[coll_a]["bm25"], cached_instance)

        # Verify new item is searchable via BM25
        results = search_bm25("verify otp", repo_id=self.repo_a)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["metadata"]["name"], "verify_otp")

    def test_bm25_cache_rebuild_after_restart(self):
        """Simulate server restart by clearing in-memory cache and verifying lazy reload."""
        store_code_chunks(self.chunks_a, repo_id=self.repo_a)
        coll_a = get_collection_name(self.repo_a)

        # Clear memory cache (simulating process restart)
        _BM25_CACHE.clear()
        self.assertNotIn(coll_a, _BM25_CACHE)

        # Search should lazily restore the cache from ChromaDB
        results = search_bm25("login authentication", repo_id=self.repo_a)
        self.assertGreater(len(results), 0)
        self.assertIn(coll_a, _BM25_CACHE)

    def test_test_aware_retrieval(self):
        """Implementation questions should prioritize source files; test questions prioritize tests."""
        store_code_chunks(self.chunks_a, repo_id=self.repo_a)

        # Implementation query: should prefer auth/service.py over tests/test_auth.py
        impl_results = search_hybrid("Where is login_user implemented?", repo_id=self.repo_a, n_results=2)
        self.assertGreater(len(impl_results), 0)
        self.assertEqual(impl_results[0]["metadata"]["file_path"], "auth/service.py")

        # Explicit test query: should prefer tests/test_auth.py
        test_results = search_hybrid("Where is the unit test for login_user?", repo_id=self.repo_a, n_results=2)
        self.assertGreater(len(test_results), 0)
        self.assertEqual(test_results[0]["metadata"]["file_path"], "tests/test_auth.py")

    def test_intent_detection(self):
        """Helper to verify explicit test intent matching."""
        self.assertTrue(is_explicit_test_query("Show me the unit test for parser"))
        self.assertTrue(is_explicit_test_query("Which tests cover chunker?"))
        self.assertTrue(is_explicit_test_query("Where is the assertion for invalid URL?"))
        self.assertFalse(is_explicit_test_query("Where is the health check endpoint implemented?"))
        self.assertFalse(is_explicit_test_query("Which function handles repository scanning?"))


if __name__ == "__main__":
    unittest.main()
