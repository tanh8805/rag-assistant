from typing import Literal, get_args

from sqlalchemy.orm import Session

from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .fts_retriever import FTSRetriever
from .rrf_retriever import RRFRetriever

RetrieverName = Literal["dense", "bm25", "fts", "rrf"]
RETRIEVERS = get_args(RetrieverName)


def get_retriever(name: RetrieverName, db: Session, document_ids: list[int] | None = None) -> BaseRetriever:
    if name == "dense":
        return DenseRetriever(db, document_ids)
    if name == "bm25":
        return BM25Retriever(db, document_ids)
    if name == "fts":
        return FTSRetriever(db, document_ids)
    if name == "rrf":
        return RRFRetriever(DenseRetriever(db, document_ids), BM25Retriever(db, document_ids))
    raise ValueError(f"Không hỗ trợ retriever '{name}'. Hỗ trợ: {', '.join(RETRIEVERS)}")
