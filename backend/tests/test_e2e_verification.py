import unittest
import py_compile
from fastapi.testclient import TestClient
from app.main import app


class TestEndToEndWorkflows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_home(self):
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json(), {"status": "healthy"})

        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertEqual(r_home.json()["status"], "running")

    def test_02_scan_validation_and_indexing(self):
        # 1. Invalid URL check
        r_bad = self.client.post("/repository/scan", json={"github_url": "http://insecure.com/repo"})
        self.assertEqual(r_bad.status_code, 400)

        # 2. Scan safe existing local checkout of CodeSage
        r_scan = self.client.post("/repository/scan", json={"github_url": "https://github.com/Payal-03/CodeSage"})
        self.assertEqual(r_scan.status_code, 200)
        data = r_scan.json()
        self.assertGreater(data.get("total_files", 0), 0)
        self.assertGreater(data.get("total_chunks", 0), 0)

    def test_03_ask_with_sources(self):
        q = "Where is the health check endpoint implemented?"
        r_ask = self.client.post("/repository/ask", json={"question": q})
        self.assertEqual(r_ask.status_code, 200)
        data = r_ask.json()
        self.assertIn("question", data)
        self.assertIn("answer", data)
        self.assertIn("sources", data)

        sources = data["sources"]
        self.assertGreater(len(sources), 0)
        top_src = sources[0]
        self.assertIn("file_path", top_src)
        self.assertIn("start_line", top_src)
        self.assertIn("end_line", top_src)
        self.assertIn("name", top_src)

    def test_04_investigate_bug(self):
        payload = {
            "bug_description": "ValueError thrown when repository path does not exist",
            "error_trace": "ValueError: Invalid repository directory"
        }
        r_bug = self.client.post("/repository/investigate-bug", json=payload)
        self.assertEqual(r_bug.status_code, 200)
        data = r_bug.json()
        self.assertIn("bug_description", data)
        self.assertIn("diagnosis", data)
        self.assertIn("affected_files", data)
        self.assertIn("sources", data)

    def test_05_streamlit_frontend_compilation(self):
        # Verifies Streamlit app has zero syntax or compilation errors
        py_compile.compile("frontend/streamlit_app.py", doraise=True)


if __name__ == "__main__":
    unittest.main()

