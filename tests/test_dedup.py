"""ingest_file bỏ qua file đã có (cùng hash nội dung). Cần database như test_retrieval."""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import src.embeddings.embedder as embedder
from src.config import settings
from src.ingestion.pipeline import ingest_file



def fake_embed(texts):
    return [[1.0] + [0.0] * (settings.embedding.dim - 1) for _ in texts]


CONTENT = "Mèo thích ngủ. Mèo bắt chuột giỏi. Mèo con cần bú sữa mẹ."


@pytest.fixture(params=["parent_child", "flat"])
def strategy(request):
    return request.param


@pytest.fixture
def db(strategy, tmp_path):
    url = os.getenv(f"TEST_DATABASE_URL_{strategy.upper()}", settings.database_url_for(strategy))
    engine = create_engine(url)
    session = Session(engine)
    try:
        session.execute(text("SELECT content_hash FROM documents LIMIT 1"))
    except Exception:
        pytest.skip(f"Không kết nối được database của strategy {strategy}")
    yield session
    session.rollback()
    session.execute(text("DELETE FROM documents WHERE source LIKE :p"), {"p": str(tmp_path) + "%"})
    session.commit()
    session.close()
    engine.dispose()


def count_docs(db, tmp_path):
    q = text("SELECT count(*) FROM documents WHERE source LIKE :p")
    return db.execute(q, {"p": str(tmp_path) + "%"}).scalar_one()


def test_ingest_skips_same_content(db, strategy, tmp_path, monkeypatch):
    monkeypatch.setattr(embedder, "embed_texts", fake_embed)
    a, copy = tmp_path / "a.txt", tmp_path / "copy.txt"
    a.write_text(CONTENT, encoding="utf-8")
    copy.write_text(CONTENT, encoding="utf-8")

    assert ingest_file(db, a, strategy) is not None
    assert ingest_file(db, a, strategy) is None  # cùng file
    assert ingest_file(db, copy, strategy) is None  # tên khác, nội dung giống
    assert count_docs(db, tmp_path) == 1


def test_ingest_again_when_content_changes(db, strategy, tmp_path, monkeypatch):
    monkeypatch.setattr(embedder, "embed_texts", fake_embed)
    a = tmp_path / "a.txt"
    a.write_text(CONTENT, encoding="utf-8")
    ingest_file(db, a, strategy)

    a.write_text(CONTENT + " Thêm một câu mới.", encoding="utf-8")
    assert ingest_file(db, a, strategy) is not None
    assert count_docs(db, tmp_path) == 1  # thay bản cũ, không nhân đôi
