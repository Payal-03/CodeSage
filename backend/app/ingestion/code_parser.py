
from tree_sitter_language_pack import get_parser


def parse_python_code(code: str):
    parser = get_parser("python")
    tree = parser.parse(bytes(code, "utf8"))

    results = []

    def traverse(node):
        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")

            if name_node:
                results.append({
                    "type": (
                        "function"
                        if node.type == "function_definition"
                        else "class"
                    ),
                    "name": code[name_node.start_byte:name_node.end_byte],
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                })

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return results
