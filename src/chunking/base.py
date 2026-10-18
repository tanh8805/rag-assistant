from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion import Document

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def make_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS)


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
    @abstractmethod
    def split(self, doc: Document) -> list[ParentChunk]:
        pass
