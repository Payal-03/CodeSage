
from app.ingestion.code_parser import parse_python_code

sample_code = """
class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return "Hello " + self.name

def add(a, b):
    return a + b
"""

results = parse_python_code(sample_code)

for item in results:
    print(item)
