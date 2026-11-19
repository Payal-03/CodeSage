from pathlib import Path
import json
from tree_sitter_language_pack import get_parser, configure, PackConfig

# Initialize local parser cache if available so parsing works offline / in sandboxes
LOCAL_PARSER_DIR = Path(__file__).resolve().parent / "parsers"
if LOCAL_PARSER_DIR.exists():
    try:
        cfg_path = str(LOCAL_PARSER_DIR).replace("\\", "/")
        configure(PackConfig.from_json(json.dumps({"cache_dir": cfg_path})))
    except Exception:
        pass


def _extract_identifiers(node, code: str) -> list[str]:
    """Extract identifier names from an assignment left-hand side pattern."""
    if node.type == "identifier":
        return [code[node.start_byte:node.end_byte]]

    names = []
    for child in node.children:
        if child.type == "identifier":
            names.append(code[child.start_byte:child.end_byte])
        elif child.type in ("pattern_list", "tuple_pattern", "list_pattern"):
            names.extend(_extract_identifiers(child, code))
    return names


def parse_python_code(code: str):
    """
    Parse Python source code using Tree-Sitter and extract structured symbols:
    - Imports (import and from ... import)
    - Module-level variables and constants (assignments, type-annotated assignments, unpacking)
    - Functions (standard and async, including decorators)
    - Classes (including decorators)
    - Methods (with qualified names like ClassName.method_name and decorators)
    """
    if not code or not code.strip():
        return []

    try:
        parser = get_parser("python")
        tree = parser.parse(bytes(code, "utf8"))
    except Exception as e:
        # Fallback empty if parser fails
        return []

    results = []

    def process_function(node, enclosing_class: str = None, outer_node = None):
        """Process function_definition or async_function_definition node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        raw_name = code[name_node.start_byte:name_node.end_byte]
        target_node = outer_node if outer_node is not None else node

        if enclosing_class:
            symbol_type = "method"
            symbol_name = f"{enclosing_class}.{raw_name}"
        else:
            symbol_type = "function"
            symbol_name = raw_name

        results.append({
            "type": symbol_type,
            "name": symbol_name,
            "start_line": target_node.start_point[0] + 1,
            "end_line": target_node.end_point[0] + 1,
            "content": code[target_node.start_byte:target_node.end_byte]
        })

    def process_class(node, outer_node = None):
        """Process class_definition node and its contained methods."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = code[name_node.start_byte:name_node.end_byte]
        target_node = outer_node if outer_node is not None else node

        results.append({
            "type": "class",
            "name": class_name,
            "start_line": target_node.start_point[0] + 1,
            "end_line": target_node.end_point[0] + 1,
            "content": code[target_node.start_byte:target_node.end_byte]
        })

        # Process class body to extract methods
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "decorated_definition":
                    inner_def = child.child_by_field_name("definition")
                    if inner_def and inner_def.type in ("function_definition", "async_function_definition"):
                        process_function(inner_def, enclosing_class=class_name, outer_node=child)
                elif child.type in ("function_definition", "async_function_definition"):
                    process_function(child, enclosing_class=class_name)

    def traverse(node, is_module_level=True):
        node_type = node.type

        # Handle decorated definitions (functions, async functions, classes with decorators)
        if node_type == "decorated_definition":
            inner_def = node.child_by_field_name("definition")
            if inner_def:
                if inner_def.type in ("function_definition", "async_function_definition"):
                    process_function(inner_def, enclosing_class=None, outer_node=node)
                    return
                elif inner_def.type == "class_definition":
                    process_class(inner_def, outer_node=node)
                    return

        # Top-level functions (without decorators)
        elif node_type in ("function_definition", "async_function_definition"):
            process_function(node, enclosing_class=None)
            return

        # Top-level classes (without decorators)
        elif node_type == "class_definition":
            process_class(node)
            return

        # Module-level variables and constants
        elif is_module_level and node_type in ("assignment", "annotated_assignment"):
            left_node = node.child_by_field_name("left")
            if left_node:
                var_names = _extract_identifiers(left_node, code)
                for var_name in var_names:
                    results.append({
                        "type": "variable",
                        "name": var_name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": code[node.start_byte:node.end_byte]
                    })
            return

        # Imports
        elif is_module_level and node_type in ("import_statement", "import_from_statement"):
            stmt_text = code[node.start_byte:node.end_byte].strip()
            results.append({
                "type": "import",
                "name": stmt_text,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "content": code[node.start_byte:node.end_byte]
            })
            return

        # Do not recurse into nested scopes for module-level variables/imports
        child_is_module_level = (
            is_module_level
            and node_type not in (
                "function_definition",
                "async_function_definition",
                "class_definition",
                "decorated_definition"
            )
        )

        for child in node.children:
            if child.is_named:
                traverse(child, child_is_module_level)

    traverse(tree.root_node, is_module_level=True)

    return results
