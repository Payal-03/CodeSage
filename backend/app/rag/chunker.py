
def create_code_chunks(files):
    chunks = []

    for file in files:
        file_path = file["file_path"]
        content = file["content"]
        symbols = file["symbols"]

        for symbol in symbols:
            start = symbol["start_line"]
            end = symbol["end_line"]

            lines = content.splitlines()
            code = "\n".join(lines[start - 1:end])

            chunks.append({
                "file_path": file_path,
                "name": symbol["name"],
                "type": symbol["type"],
                "start_line": start,
                "end_line": end,
                "content": code
            })

    return chunks
