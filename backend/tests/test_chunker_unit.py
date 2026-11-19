import unittest
from app.rag.chunker import create_code_chunks


class TestChunkerUnit(unittest.TestCase):

    def test_empty_files(self):
        chunks = create_code_chunks([])
        self.assertEqual(chunks, [])

    def test_consolidate_imports(self):
        files = [{
            "file_path": "sample.py",
            "content": "import os\nimport sys\nfrom math import sqrt\n",
            "total_lines": 3,
            "symbols": [
                {"type": "import", "name": "import os", "start_line": 1, "end_line": 1, "content": "import os"},
                {"type": "import", "name": "import sys", "start_line": 2, "end_line": 2, "content": "import sys"},
                {"type": "import", "name": "from math import sqrt", "start_line": 3, "end_line": 3, "content": "from math import sqrt"}
            ]
        }]
        chunks = create_code_chunks(files)
        # Should be consolidated into 1 chunk for imports
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "imports")
        self.assertEqual(chunks[0]["name"], "sample_imports")
        self.assertEqual(chunks[0]["start_line"], 1)
        self.assertEqual(chunks[0]["end_line"], 3)
        self.assertIn("import os", chunks[0]["content"])
        self.assertIn("from math import sqrt", chunks[0]["content"])

    def test_functions_and_classes_chunking(self):
        files = [{
            "file_path": "calc.py",
            "content": "def add(a, b):\n    return a + b\n\nclass MathCalc:\n    pass\n",
            "total_lines": 5,
            "symbols": [
                {"type": "function", "name": "add", "start_line": 1, "end_line": 2, "content": "def add(a, b):\n    return a + b"},
                {"type": "class", "name": "MathCalc", "start_line": 4, "end_line": 5, "content": "class MathCalc:\n    pass"}
            ]
        }]
        chunks = create_code_chunks(files)
        self.assertEqual(len(chunks), 2)
        types = [c["type"] for c in chunks]
        names = [c["name"] for c in chunks]
        self.assertIn("function", types)
        self.assertIn("class", types)
        self.assertIn("add", names)
        self.assertIn("MathCalc", names)

    def test_fallback_for_symbol_less_file(self):
        markdown_content = "# Project Documentation\n\nThis is a sample readme."
        files = [{
            "file_path": "README.md",
            "content": markdown_content,
            "total_lines": 3,
            "symbols": []
        }]
        chunks = create_code_chunks(files)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "file_block")
        self.assertEqual(chunks[0]["file_path"], "README.md")
        self.assertIn("Project Documentation", chunks[0]["content"])


if __name__ == "__main__":
    unittest.main()

