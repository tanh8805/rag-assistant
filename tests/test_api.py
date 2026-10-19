"""Test /ingest và /query bằng TestClient: embedding và LLM giả, database thật (parent_child).
Không kết nối được database thì test tự bỏ qua."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import src.api.ingest as ingest_api
import src.api.query as query_api
import src.embeddings.embedder as embedder
import src.retrieval.dense_retriever as dense_module
from src.config import settings
from src.db import get_db
from src.generation import BaseLLM
from src.main import app

CONTENT = "Mèo thích ngủ nhiều. Mèo bắt chuột rất giỏi. Mèo con cần bú sữa mẹ."


def fake_embed(texts):
    return [[1.0] + [0.0] * (settings.embedding.dim - 1) for _ in texts]


class FakeLLM(BaseLLM):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return "câu trả lời giả"


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = os.getenv("TEST_DATABASE_URL_PARENT_CHILD", settings.database_url_for("parent_child"))
    engine = create_engine(url)
    db = Session(engine)
    try:
        db.execute(text("SELECT content_hash FROM documents LIMIT 1"))
    except Exception:
        pytest.skip("Không kết nối được database parent_child")

    llm = FakeLLM()
    monkeypatch.setattr(embedder, "embed_texts", fake_embed)  # dùng khi ingest
    monkeypatch.setattr(dense_module, "embed_texts", fake_embed)  # dùng khi query
    monkeypatch.setattr(query_api, "rerank", lambda q, parents: parents[: settings.rerank.top_n])  # bỏ model rerank thật
    monkeypatch.setattr(ingest_api, "UPLOAD_DIR", tmp_path / "uploads")
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[query_api.default_llm] = lambda: llm
    yield TestClient(app), llm, tmp_path
    app.dependency_overrides.clear()
    db.rollback()
    db.execute(text("DELETE FROM documents WHERE source LIKE :p"), {"p": str(tmp_path) + "%"})
    db.commit()
    db.close()
    engine.dispose()


def upload(client, name, tmp_path):
    content = f"{CONTENT} {tmp_path.name}"  # mỗi test một nội dung riêng, không trùng hash với dữ liệu có sẵn
    return client.post("/ingest", files={"file": (name, content.encode("utf-8"))})


def test_ingest_then_duplicate(env):
    client, _, tmp_path = env
    r = upload(client, "meo.txt", tmp_path)
    assert r.status_code == 200
    assert r.json()["status"] == "ingested" and r.json()["document_id"] is not None
    assert (tmp_path / "uploads" / "meo.txt").exists()

    r = upload(client, "copy.txt", tmp_path)  # tên khác, nội dung giống
    assert r.json() == {"filename": "copy.txt", "status": "duplicate", "document_id": None}
    assert not (tmp_path / "uploads" / "copy.txt").exists()


def test_ingest_rejects_unsupported_format(env):
    client, _, tmp_path = env
    assert upload(client, "a.exe", tmp_path).status_code == 400


@pytest.mark.parametrize("retriever", ["dense", "bm25", "fts", "rrf"])
def test_query_returns_answer_and_sources(env, retriever):
    client, llm, tmp_path = env
    document_id = upload(client, "meo.txt", tmp_path).json()["document_id"]

    r = client.post(
        "/query", json={"question": "Mèo thích gì?", "retriever": retriever, "document_ids": [document_id]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "câu trả lời giả"
    assert body["sources"] and "Mèo" in body["sources"][0]["content"]
    assert "Mèo thích gì?" in llm.prompts[-1] and "Mèo thích ngủ" in llm.prompts[-1]


def test_query_without_matches_skips_llm(env):
    client, llm, _ = env
    r = client.post("/query", json={"question": "Mèo thích gì?", "retriever": "dense", "document_ids": []})
    assert r.status_code == 200
    assert r.json()["sources"] == []
    assert llm.prompts == []


def test_query_validates_request(env):
    client, _, _ = env
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "retriever": "abc"}).status_code == 422


def test_query_without_api_key_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    query_api.default_llm.cache_clear()
    r = TestClient(app).post("/query", json={"question": "xin chào"})
    assert r.status_code == 503
    assert "API_KEY" in r.json()["detail"]
