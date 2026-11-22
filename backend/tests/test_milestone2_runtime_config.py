import unittest
import os
from unittest.mock import patch, MagicMock

from app.llm import get_ollama_url, get_ollama_model, ask_llm
from app.main import app, home, health


class TestMilestone2RuntimeConfig(unittest.TestCase):

    def setUp(self):
        # Save original environment
        self._orig_url = os.environ.get("OLLAMA_URL")
        self._orig_model = os.environ.get("OLLAMA_MODEL")

    def tearDown(self):
        # Restore environment
        if self._orig_url is not None:
            os.environ["OLLAMA_URL"] = self._orig_url
        else:
            os.environ.pop("OLLAMA_URL", None)

        if self._orig_model is not None:
            os.environ["OLLAMA_MODEL"] = self._orig_model
        else:
            os.environ.pop("OLLAMA_MODEL", None)

    def test_default_ollama_config(self):
        """Verifies default Ollama endpoint and model when environment variables are omitted."""
        os.environ.pop("OLLAMA_URL", None)
        os.environ.pop("OLLAMA_MODEL", None)

        self.assertEqual(get_ollama_url(), "http://127.0.0.1:11434/api/generate")
        self.assertEqual(get_ollama_model(), "qwen2.5:3b")

    def test_custom_ollama_config(self):
        """Verifies custom Ollama endpoint and model override via environment variables."""
        custom_url = "http://remote-server:11434/api/generate"
        custom_model = "qwen2.5-coder:7b"

        os.environ["OLLAMA_URL"] = custom_url
        os.environ["OLLAMA_MODEL"] = custom_model

        self.assertEqual(get_ollama_url(), custom_url)
        self.assertEqual(get_ollama_model(), custom_model)

        # Mock requests.post to verify ask_llm uses the custom values in HTTP payload
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "Mocked response from custom LLM"}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            answer = ask_llm("Test question")

            mock_post.assert_called_once()
            called_url, called_kwargs = mock_post.call_args
            self.assertEqual(called_url[0], custom_url)
            self.assertEqual(called_kwargs["json"]["model"], custom_model)
            self.assertEqual(called_kwargs["json"]["prompt"], "Test question")
            self.assertEqual(answer, "Mocked response from custom LLM")

    def test_fast_health_and_home_execution(self):
        """Health and home endpoints should execute instantly without ML overhead."""
        res_home = home()
        self.assertEqual(res_home["status"], "running")
        self.assertIn("Welcome", res_home["message"])

        res_health = health()
        self.assertEqual(res_health, {"status": "healthy"})


if __name__ == "__main__":
    unittest.main()
