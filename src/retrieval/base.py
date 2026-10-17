from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings


@dataclass
class RetrievedChunk:
    chunk_id: int
    parent_id: int
    content: str
    score: float


@dataclass
class RetrievedParent:
    parent_id: int
    content: str
    source: str


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        pass


def get_parents(db: Session, chunks: list[RetrievedChunk]) -> list[RetrievedParent]:
    parent_ids = list(dict.fromkeys(c.parent_id for c in chunks))
    rows = db.execute(
        text(
            "SELECT p.id, p.content, d.source FROM parents p "
            "JOIN documents d ON d.id = p.document_id WHERE p.id = ANY(:ids)"
        ),
        {"ids": parent_ids},
    ).all()
    by_id = {r.id: RetrievedParent(r.id, r.content, r.source) for r in rows}
    return [by_id[pid] for pid in parent_ids]
