from langsmith import traceable
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings

from .base import BaseRetriever, RetrievedChunk
from .bm25_retriever import tokenize


class FTSRetriever(BaseRetriever):
    """Tìm theo từ khóa bằng PostgreSQL Full-Text Search, dùng GIN index idx_chunks_fts."""

    def __init__(self, db: Session, document_ids: list[int] | None = None):
        self.db = db
        self.document_ids = document_ids

    @traceable(run_type="retriever", name="fts_retrieve")
    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        params = {"q": " | ".join(tokens), "k": top_k}  
        where = ""
        if self.document_ids is not None:
            where = "AND c.document_id = ANY(:doc_ids)"
            params["doc_ids"] = self.document_ids

        rows = self.db.execute(
            text(
                "SELECT c.id, c.content, ts_rank(c.content_tsv, q) AS score "
                "FROM chunks c, to_tsquery('simple', :q) q "
                f"WHERE c.content_tsv @@ q {where} "
                "ORDER BY score DESC LIMIT :k"
            ),
            params,
        ).all()
        return [RetrievedChunk(r.id, r.content, r.score) for r in rows]
