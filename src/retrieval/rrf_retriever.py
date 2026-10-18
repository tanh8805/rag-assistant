from src.config import settings

from .base import BaseRetriever, RetrievedChunk
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

RRF_K = 60


class RRFRetriever(BaseRetriever):

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever):
        self.dense = dense
        self.bm25 = bm25

    def retrieve(self, query: str, top_k: int = settings.retrieval.top_k) -> list[RetrievedChunk]:
        scores: dict[int, float] = {}
        chunks: dict[int, RetrievedChunk] = {}

        for results in (self.dense.retrieve(query, top_k), self.bm25.retrieve(query, top_k)):
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
                chunks[chunk.chunk_id] = chunk

        best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [RetrievedChunk(cid, chunks[cid].content, scores[cid]) for cid in best]
