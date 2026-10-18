import re


def norm(s: str) -> str:
    """Gộp khoảng trắng/xuống dòng để so khớp."""
    return re.sub(r"\s+", " ", s).strip()


def recall_at_k(ranked: list[str], spans: list[str], k: int) -> float:
    """Tỉ lệ đoạn bằng chứng (span) nằm trong top-k văn bản tìm được."""
    top = [norm(t) for t in ranked[:k]]
    found = sum(any(norm(s) in t for t in top) for s in spans)
    return found / len(spans)


def reciprocal_rank(ranked: list[str], spans: list[str]) -> float:
    """1 / thứ hạng của văn bản đầu tiên chứa một span bất kỳ, 0 nếu không có."""
    spans = [norm(s) for s in spans]
    for rank, text in enumerate(ranked, start=1):
        text = norm(text)
        if any(s in text for s in spans):
            return 1 / rank
    return 0.0
