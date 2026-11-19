import unittest
from app.ingestion.code_parser import parse_python_code


class TestCodeParser(unittest.TestCase):

    def test_empty_and_whitespace(self):
        self.assertEqual(parse_python_code(""), [])
        self.assertEqual(parse_python_code("   \n\n\t  "), [])

    def test_imports_parsing(self):
        code = "import os\nfrom pathlib import Path, PurePath\n"
        symbols = parse_python_code(code)
        import_symbols = [s for s in symbols if s["type"] == "import"]
        self.assertEqual(len(import_symbols), 2)
        self.assertIn("import os", import_symbols[0]["name"])
        self.assertIn("from pathlib import Path, PurePath", import_symbols[1]["name"])

    def test_module_variables_and_unpacking(self):
        code = """
API_KEY = "12345"
TIMEOUT: int = 30
HOST, PORT = "0.0.0.0", 8080

def test_func():
    inner_var = 1
"""
        symbols = parse_python_code(code)
        var_symbols = [s for s in symbols if s["type"] == "variable"]
        var_names = [v["name"] for v in var_symbols]

        self.assertIn("API_KEY", var_names)
        self.assertIn("TIMEOUT", var_names)
        self.assertIn("HOST", var_names)
        self.assertIn("PORT", var_names)
        # Inner local variable inside function must NOT be extracted
        self.assertNotIn("inner_var", var_names)

    def test_functions_and_async(self):
        code = """
def normal_sync(a, b):
    return a + b

async def fetch_remote():
    return True
"""
        symbols = parse_python_code(code)
        func_symbols = [s for s in symbols if s["type"] == "function"]
        func_names = [f["name"] for f in func_symbols]

        self.assertIn("normal_sync", func_names)
        self.assertIn("fetch_remote", func_names)

    def test_decorated_functions_and_classes(self):
        code = """
@app.get("/items")
async def get_items():
    return []

@dataclass
class UserRecord:
    name: str

    @property
    def display_name(self) -> str:
        return self.name
"""
        symbols = parse_python_code(code)
        symbols_by_name = {s["name"]: s for s in symbols}

        # Decorated function preserves decorator in snippet
        self.assertIn("get_items", symbols_by_name)
        self.assertIn('@app.get("/items")', symbols_by_name["get_items"]["content"])

        # Decorated class preserves decorator
        self.assertIn("UserRecord", symbols_by_name)
        self.assertEqual(symbols_by_name["UserRecord"]["type"], "class")
        self.assertIn("@dataclass", symbols_by_name["UserRecord"]["content"])

        # Method with qualified name
        self.assertIn("UserRecord.display_name", symbols_by_name)
        self.assertEqual(symbols_by_name["UserRecord.display_name"]["type"], "method")
        self.assertIn("@property", symbols_by_name["UserRecord.display_name"]["content"])


if __name__ == "__main__":
    unittest.main()

