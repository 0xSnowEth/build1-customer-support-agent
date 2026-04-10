def read_document(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def answer_question(document: str, question: str) -> str:
    # This one Claude handles internally
    # Just return both combined as context
    return f"DOCUMENT:\n{document}\n\nQUESTION:\n{question}"