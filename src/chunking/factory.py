from src.config import settings

from .base import BaseChunker
from .recursive_chunker import RecursiveChunker

CHUNKERS: dict[str, type[BaseChunker]] = {
    "recursive": RecursiveChunker,
}


def get_chunker(strategy: str | None = None) -> BaseChunker:
    c = settings.chunking
    strategy = strategy or c.strategy
    if strategy not in CHUNKERS:
        raise ValueError(f"Không hỗ trợ chunking strategy '{strategy}'. Hỗ trợ: {', '.join(CHUNKERS)}")
    return CHUNKERS[strategy](c.parent_size, c.parent_overlap, c.child_size, c.child_overlap)
