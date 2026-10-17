"""Cần database đang chạy (docker compose up db). Không kết nối được thì test tự bỏ qua.
Dùng embedding giả (không tải model) và chỉ tìm trong các tài liệu test tạo ra."""

import hashlib
import math
import os
import re

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import src.embeddings.embedder as embedder
import src.retrieval.dense_retriever as dense_module
from src.chunking import RecursiveChunker
from src.config import settings
from src.db import save_document
from src.ingestion import Document
from src.retrieval import BM25Retriever, DenseRetriever, RRFRetriever, get_parents

PREFIX = "test_retrieval/"
DOCS = {
    "meo": "Mèo là loài động vật ăn thịt. Mèo thích ngủ nhiều và bắt chuột rất giỏi. "
    "Mèo con cần bú sữa mẹ. Lông mèo mềm và có nhiều màu sắc khác nhau.",
    "python": "Python là ngôn ngữ lập trình phổ biến. Python dùng nhiều trong trí tuệ nhân tạo. "
    "Lập trình viên thích Python vì cú pháp đơn giản. Python có thư viện rất phong phú.",
    "pho": "Phở là món ăn truyền thống của Hà Nội. Nước dùng phở được ninh từ xương bò. "
    "Bánh phở mềm ăn kèm thịt bò và hành lá. Người Hà Nội thường ăn phở vào buổi sáng.",
}


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for t in texts:
        v = [0.0] * settings.embedding.dim
        for word in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % len(v)] += 1
        norm = math.sqrt(sum(x * x for x in v)) or 1
        vectors.append([x / norm for x in v])
    return vectors


@pytest.fixture(scope="module")
def db():
    engine = create_engine(os.getenv("TEST_DATABASE_URL", settings.database_url))
    session = Session(engine)
    try:
        session.execute(text("SELECT 1 FROM chunks LIMIT 1"))
    except Exception:
        pytest.skip("Không kết nối được database")
    yield session
    session.rollback()
    session.execute(text("DELETE FROM documents WHERE source LIKE :p"), {"p": PREFIX + "%"})
    session.commit()
    session.close()
    engine.dispose()


@pytest.fixture(scope="module")
def doc_ids(db):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(embedder, "embed_texts", fake_embed)
        mp.setattr(dense_module, "embed_texts", fake_embed)

        chunker = RecursiveChunker(parent_size=500, parent_overlap=0, child_size=100, child_overlap=0)
        ids = {}
        for name, content in DOCS.items():
            source = f"{PREFIX}{name}.txt"
            parents = chunker.split(Document(content, source, {"title": name}))
            embedder.embed_parents(parents)
            ids[name] = save_document(db, source, name, parents)
        yield ids


@pytest.fixture
def retrievers(db, doc_ids):
    ids = list(doc_ids.values())
    dense, bm25 = DenseRetriever(db, ids), BM25Retriever(db, ids)
    return {"dense": dense, "bm25": bm25, "rrf": RRFRetriever(dense, bm25)}


@pytest.fixture(autouse=True)
def use_fake_embedding(monkeypatch):
    monkeypatch.setattr(dense_module, "embed_texts", fake_embed)


@pytest.mark.parametrize("name", ["dense", "bm25", "rrf"])
@pytest.mark.parametrize(
    "query, expected",
    [("mèo bắt chuột", "meo"), ("lập trình Python", "python"), ("phở Hà Nội", "pho")],
)
def test_top_result_is_in_right_document(db, retrievers, name, query, expected):
    hits = retrievers[name].retrieve(query, top_k=3)
    assert get_parents(db, hits)[0].source == f"{PREFIX}{expected}.txt"


@pytest.mark.parametrize("name", ["dense", "bm25", "rrf"])
def test_top_k_and_sorted_by_score(retrievers, name):
    hits = retrievers[name].retrieve("là", top_k=3)
    assert 0 < len(hits) <= 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_returns_nothing_without_matching_word(retrievers):
    assert retrievers["bm25"].retrieve("zzzz qqqq") == []


def test_rrf_has_no_duplicate_chunks(retrievers):
    ids = [h.chunk_id for h in retrievers["rrf"].retrieve("là", top_k=10)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("make", [DenseRetriever, BM25Retriever])
def test_document_filter(db, doc_ids, make):
    retriever = make(db, [doc_ids["meo"]])
    parents = get_parents(db, retriever.retrieve("lập trình Python là gì", top_k=10))
    assert parents
    assert {p.source for p in parents} == {f"{PREFIX}meo.txt"}


@pytest.mark.parametrize("make", [DenseRetriever, BM25Retriever])
def test_empty_document_ids_returns_nothing(db, doc_ids, make):
    assert make(db, []).retrieve("mèo") == []


def test_get_parents_removes_duplicates(db, retrievers):
    hits = retrievers["dense"].retrieve("Python", top_k=10)
    parents = get_parents(db, hits)
    assert len(parents) < len(hits)  
    assert len({p.parent_id for p in parents}) == len(parents)
