from src.config import settings

from .base import BaseChunker
from .flat_chunker import FlatChunker
from .parent_child_chunker import ParentChildChunker


def get_chunker(strategy: str | None = None) -> BaseChunker:
    c = settings.chunking
    strategy = strategy or c.strategy
    if strategy == "parent_child":
        return ParentChildChunker(c.parent_size, c.child_size, c.child_overlap)
    if strategy == "flat":
        return FlatChunker(c.chunk_size, c.chunk_overlap)
    raise ValueError(f"Không hỗ trợ chunking strategy '{strategy}'. Hỗ trợ: parent_child, flat")
