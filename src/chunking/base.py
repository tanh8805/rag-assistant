from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.ingestion import Document


@dataclass
class ChildChunk:
    content: str
    parent_index: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class ParentChunk:
    content: str
    source: str  
    children: list[ChildChunk]
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    def __init__(self, parent_size: int, parent_overlap: int, child_size: int, child_overlap: int):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    @abstractmethod
    def split(self, doc: Document) -> list[ParentChunk]:
        pass
