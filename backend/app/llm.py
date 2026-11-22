import os
import requests


def get_ollama_url() -> str:
    """Returns Ollama generation endpoint from environment or default."""
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")


def get_ollama_model() -> str:
    """Returns Ollama model name from environment or default."""
    return os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


# Backwards compatibility module attributes
OLLAMA_URL = get_ollama_url()
MODEL_NAME = get_ollama_model()


def ask_llm(question: str):
    """
    Queries local Ollama instance running the configured model (default: Qwen 2.5 3B).
    Uses separate connect and read timeouts for responsive failure handling.
    """
    url = get_ollama_url()
    model = get_ollama_model()

    payload = {
        "model": model,
        "prompt": question,
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(5, 120)  # 5s connect timeout, 120s generation timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to Ollama at {url}. "
            f"Please ensure Ollama is running (`ollama serve`) and model '{model}' is loaded."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama generation timed out after 120 seconds.")
