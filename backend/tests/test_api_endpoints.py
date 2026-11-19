import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestApiEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "healthy"})

    def test_home_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("message", res.json())
        self.assertEqual(res.json()["status"], "running")

    def test_invalid_github_url_validation(self):
        res = self.client.post("/repository/scan", json={"github_url": "invalid_url"})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()

