"""Test rerank bằng cross-encoder giả (không tải model thật)."""

import src.rerank.reranker as reranker
from src.retrieval import RetrievedParent


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=32):
        return [len(text) for _, text in pairs]  # văn bản càng dài điểm càng cao


def make_parents(*contents):
    return [RetrievedParent(parent_id=i, content=c, source="a.txt") for i, c in enumerate(contents)]


def test_rerank_sorts_by_score_and_keeps_top_n(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeCrossEncoder())
    parents = make_parents("ngắn", "dài hơn một chút", "dài nhất trong ba văn bản này")
    result = reranker.rerank("câu hỏi", parents, top_n=2)
    assert [p.parent_id for p in result] == [2, 1]


def test_rerank_empty(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeCrossEncoder())
    assert reranker.rerank("câu hỏi", []) == []
