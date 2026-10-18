from src.ingestion import Document

from .base import BaseChunker, ChildChunk, ParentChunk, make_splitter


class FlatChunker(BaseChunker):

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = make_splitter(chunk_size, chunk_overlap)

    def split(self, doc: Document) -> list[ParentChunk]:
        return [
            ParentChunk(text, doc.source, [ChildChunk(text, index)], dict(doc.metadata))
            for index, text in enumerate(self.splitter.split_text(doc.content))
        ]
