import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def ask_llm(question: str):
    """
    Queries local Ollama instance running Qwen 2.5 3B.
    Uses separate connect and read timeouts for responsive failure handling.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(5, 120)  # 5s connect timeout, 120s generation timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Please ensure Ollama is running (`ollama serve`) and model 'qwen2.5:3b' is loaded."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama generation timed out after 120 seconds.")
