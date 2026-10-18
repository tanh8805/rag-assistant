from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings
from src.embeddings import embed_texts

from .base import BaseRetriever, RetrievedChunk


class DenseRetriever(BaseRetriever):

    def __init__(self, db: Session, document_ids: list[int] | None = None):
        self.db = db
        self.document_ids = document_ids

    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        params = {"v": str(embed_texts([query])[0]), "k": top_k}
        where = ""
        if self.document_ids is not None:
            where = "WHERE c.document_id = ANY(:doc_ids)"
            params["doc_ids"] = self.document_ids

        rows = self.db.execute(
            text(
                "SELECT c.id, c.content, 1 - (c.embedding <=> CAST(:v AS vector)) AS score "
                f"FROM chunks c {where} "
                "ORDER BY c.embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            params,
        ).all()
        return [RetrievedChunk(r.id, r.content, r.score) for r in rows]
