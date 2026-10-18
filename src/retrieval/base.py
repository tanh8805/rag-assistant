from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import ChunkingStrategy, settings


@dataclass
class RetrievedChunk:
    chunk_id: int
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


def get_parents(
    db: Session, chunks: list[RetrievedChunk], strategy: ChunkingStrategy = settings.chunking.strategy
) -> list[RetrievedParent]:
    """Lấy nội dung đưa cho LLM từ các chunk tìm được (bỏ trùng, giữ thứ tự xếp hạng).

    parent_child: parent chứa chunk đó. flat: chính chunk đó (parent_id lúc này là id của chunk).
    """
    if strategy == "flat":
        sql = (
            "SELECT c.id AS chunk_id, c.id AS parent_id, c.content, d.source "
            "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(:ids)"
        )
    else:
        sql = (
            "SELECT c.id AS chunk_id, p.id AS parent_id, p.content, d.source "
            "FROM chunks c JOIN parents p ON p.id = c.parent_id "
            "JOIN documents d ON d.id = p.document_id WHERE c.id = ANY(:ids)"
        )
    rows = db.execute(text(sql), {"ids": [c.chunk_id for c in chunks]}).all()
    by_chunk = {r.chunk_id: r for r in rows}

    parents: dict[int, RetrievedParent] = {}
    for chunk in chunks:
        r = by_chunk[chunk.chunk_id]
        parents.setdefault(r.parent_id, RetrievedParent(r.parent_id, r.content, r.source))
    return list(parents.values())
