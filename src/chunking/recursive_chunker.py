from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion import Document

from .base import BaseChunker, ChildChunk, ParentChunk

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _make_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS
    )


class RecursiveChunker(BaseChunker):
    def __init__(self, parent_size: int, parent_overlap: int, child_size: int, child_overlap: int):
        super().__init__(parent_size, parent_overlap, child_size, child_overlap)
        self.parent_splitter = _make_splitter(parent_size, parent_overlap)
        self.child_splitter = _make_splitter(child_size, child_overlap)

    def split(self, doc: Document) -> list[ParentChunk]:
        parents = []
        for index, parent_text in enumerate(self.parent_splitter.split_text(doc.content)):
            children = [ChildChunk(text, index) for text in self.child_splitter.split_text(parent_text)]
            parents.append(ParentChunk(parent_text, doc.source, children, dict(doc.metadata)))
        return parents
