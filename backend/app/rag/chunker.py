from pathlib import Path


def is_test_file(file_path: str) -> bool:
    """Identifies test files by path and naming convention."""
    p = Path(file_path)
    name = p.name.lower()
    parts = [part.lower() for part in p.parts]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts
        or "test" in parts
        or name in ("test_parser.py", "test_chunker.py")
    )


def create_code_chunks(files: list[dict], max_fallback_lines: int = 60) -> list[dict]:
    """
    Transforms scanned repository files and parsed AST symbols into searchable code chunks.

    Handles:
    - Module-level imports (consolidated into a coherent import block per file to avoid 1-line noise)
    - Module-level variables and constants
    - Functions (sync and async, with decorators)
    - Classes (with decorators)
    - Methods (with qualified names like ClassName.method_name)
    - Files without symbols (Markdown, configs, procedural scripts) chunked by line blocks
    - Adds `is_test` metadata flag to differentiate test files from implementation files
    """
    chunks = []

    for file in files:
        file_path = file.get("file_path", "")
        content = file.get("content", "")
        symbols = file.get("symbols", [])
        total_lines = file.get("total_lines", len(content.splitlines()))
        is_test = is_test_file(file_path)

        if not content.strip():
            continue

        # If symbols were extracted by the AST parser
        if symbols:
            lines = content.splitlines()
            imports = []

            for symbol in symbols:
                sym_type = symbol.get("type")
                sym_name = symbol.get("name")
                start = symbol.get("start_line", 1)
                end = symbol.get("end_line", start)

                # Collect imports to consolidate them
                if sym_type == "import":
                    imports.append(symbol)
                    continue

                # Use parsed content if available, otherwise slice from lines
                code = symbol.get("content")
                if not code:
                    code = "\n".join(lines[max(0, start - 1):end])

                chunks.append({
                    "file_path": file_path,
                    "name": sym_name,
                    "type": sym_type,
                    "start_line": start,
                    "end_line": end,
                    "content": code,
                    "is_test": is_test
                })

            # Consolidate imports into a single contextual chunk per file
            if imports:
                min_line = min(imp["start_line"] for imp in imports)
                max_line = max(imp["end_line"] for imp in imports)

                # If imports are cleanly parsed, join statements
                import_texts = [imp["content"].strip() for imp in imports if imp.get("content")]
                import_code = "\n".join(import_texts)

                chunks.append({
                    "file_path": file_path,
                    "name": f"{Path(file_path).stem}_imports",
                    "type": "imports",
                    "start_line": min_line,
                    "end_line": max_line,
                    "content": import_code,
                    "is_test": is_test
                })

        else:
            # Fallback for files without AST symbols (e.g. .md, .json, scripts without definitions)
            lines = content.splitlines()
            if not lines:
                continue

            num_lines = len(lines)
            step = max_fallback_lines

            for i in range(0, num_lines, step):
                chunk_lines = lines[i:i + step]
                start_l = i + 1
                end_l = min(i + step, num_lines)
                chunk_code = "\n".join(chunk_lines)

                chunks.append({
                    "file_path": file_path,
                    "name": f"{Path(file_path).name} (lines {start_l}-{end_l})",
                    "type": "file_block",
                    "start_line": start_l,
                    "end_line": end_l,
                    "content": chunk_code,
                    "is_test": is_test
                })

    return chunks
