from app.ingestion.code_parser import parse_python_code

sample_code = '''
import requests
from fastapi import FastAPI, HTTPException
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME: str = "qwen2.5:3b"
HOST, PORT = "127.0.0.1", 8000

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

def ask_llm(question: str) -> str:
    answer = question
    return answer

@dataclass
class CodeAnalyzer:
    repo_path: str

    @property
    def is_valid(self) -> bool:
        return True

    def analyze(self):
        local_var = 10
        pass

    async def async_scan(self):
        pass
'''

results = parse_python_code(sample_code)

print(f"Total symbols extracted: {len(results)}\n")
for item in results:
    print(f"[{item['type'].upper()}] {item['name']} (Lines {item['start_line']}-{item['end_line']})")
    print(f"Snippet:\n{item['content'].strip()}")
    print("-" * 50)
