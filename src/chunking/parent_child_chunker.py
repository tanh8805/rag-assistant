from src.ingestion import Document

from .base import BaseChunker, ChildChunk, ParentChunk, make_splitter


class ParentChildChunker(BaseChunker):

    def __init__(self, parent_size: int, child_size: int, child_overlap: int):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_splitter = make_splitter(parent_size, 0)
        self.child_splitter = make_splitter(child_size, child_overlap)

    def split(self, doc: Document) -> list[ParentChunk]:
        parents = []
        for index, parent_text in enumerate(self.parent_splitter.split_text(doc.content)):
            children = [ChildChunk(text, index) for text in self.child_splitter.split_text(parent_text)]
            parents.append(ParentChunk(parent_text, doc.source, children, dict(doc.metadata)))
        return parents
