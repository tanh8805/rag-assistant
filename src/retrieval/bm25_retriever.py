import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Plus
from sqlalchemy import Row, text
from sqlalchemy.orm import Session

from src.config import settings

from .base import BaseRetriever, RetrievedChunk


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


@dataclass
class BM25Index:
    """Chỉ số BM25 của toàn bộ chunk trong một database, giữ trong RAM."""

    rows: Sequence[Row]  # (id, document_id, content) của từng chunk
    token_sets: list[set[str]]
    bm25: BM25Plus | None
    version: tuple  # (số chunk, id lớn nhất) lúc dựng, để biết khi nào dữ liệu đã đổi


_indexes: dict[str, BM25Index] = {}  # mỗi database một chỉ số
_lock = threading.Lock()


def get_index(db: Session) -> BM25Index:
    """Trả về chỉ số trong RAM; chỉ dựng lại khi chunk trong database thay đổi (có ingest hoặc xóa)."""
    key = str(db.get_bind().engine.url)
    version = tuple(db.execute(text("SELECT count(*), coalesce(max(id), 0) FROM chunks")).one())
    index = _indexes.get(key)
    if index is not None and index.version == version:
        return index

    with _lock:
        rows = db.execute(text("SELECT id, document_id, content FROM chunks")).all()
        corpus = [tokenize(r.content) for r in rows]
        index = BM25Index(rows, [set(t) for t in corpus], BM25Plus(corpus) if rows else None, version)
        _indexes[key] = index
        return index


class BM25Retriever(BaseRetriever):

    def __init__(self, db: Session, document_ids: list[int] | None = None):
        self.index = get_index(db)
        self.document_ids = None if document_ids is None else set(document_ids)

    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        index = self.index
        if index.bm25 is None:
            return []
        query_tokens = tokenize(query)
        scores = index.bm25.get_scores(query_tokens)
        matched = [
            i
            for i, tokens in enumerate(index.token_sets)
            if set(query_tokens) & tokens and (self.document_ids is None or index.rows[i].document_id in self.document_ids)
        ]
        best = sorted(matched, key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievedChunk(index.rows[i].id, index.rows[i].content, scores[i]) for i in best]
