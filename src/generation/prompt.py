from src.retrieval import RetrievedParent

SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp. "
    "Chỉ dùng thông tin trong phần Tài liệu. Nếu tài liệu không có thông tin, hãy nói không tìm thấy. "
    "Trả lời bằng tiếng Việt, ngắn gọn."
)


def build_prompt(question: str, parents: list[RetrievedParent]) -> str:
    context = "\n\n".join(f"[{i}] (nguồn: {p.source})\n{p.content}" for i, p in enumerate(parents, 1))
    return f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"
