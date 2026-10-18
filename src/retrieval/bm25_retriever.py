import re

from rank_bm25 import BM25Plus
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings

from .base import BaseRetriever, RetrievedChunk


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class BM25Retriever(BaseRetriever):

    def __init__(self, db: Session, document_ids: list[int] | None = None):
        query = "SELECT c.id, c.content FROM chunks c"
        params = {}
        if document_ids is not None:
            query += " WHERE c.document_id = ANY(:doc_ids)"
            params["doc_ids"] = document_ids

        self.rows = db.execute(text(query), params).all()
        self.corpus = [tokenize(r.content) for r in self.rows]
        self.bm25 = BM25Plus(self.corpus) if self.rows else None

    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        if self.bm25 is None:
            return []
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        matched = [i for i, tokens in enumerate(self.corpus) if set(query_tokens) & set(tokens)]
        best = sorted(matched, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(self.rows[i].id, self.rows[i].content, scores[i])
            for i in best
        ]
