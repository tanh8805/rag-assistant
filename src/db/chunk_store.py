import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.chunking import ParentChunk


def document_exists(db: Session, content_hash: str) -> bool:
    """Đã có tài liệu nào cùng nội dung chưa."""
    row = db.execute(
        text("SELECT 1 FROM documents WHERE content_hash = :h LIMIT 1"), {"h": content_hash}
    ).first()
    return row is not None


def _replace_document(db: Session, source: str, title: str, content_hash: str | None) -> int:
    """Xóa bản cũ cùng source  tạo document mới."""
    db.execute(text("DELETE FROM documents WHERE source = :source"), {"source": source})
    return db.execute(
        text(
            "INSERT INTO documents (source, title, content_hash) "
            "VALUES (:source, :title, :content_hash) RETURNING id"
        ),
        {"source": source, "title": title, "content_hash": content_hash},
    ).scalar_one()


def save_document(
    db: Session,
    source: str,
    title: str,
    parents: list[ParentChunk],
    content_hash: str | None = None,
) -> int:
    """Lưu tài liệu theo schema parent-child """
    try:
        document_id = _replace_document(db, source, title, content_hash)
        for parent in parents:
            parent_id = db.execute(
                text(
                    "INSERT INTO parents (document_id, content, metadata) "
                    "VALUES (:document_id, :content, CAST(:metadata AS jsonb)) RETURNING id"
                ),
                {
                    "document_id": document_id,
                    "content": parent.content,
                    "metadata": json.dumps(parent.metadata),
                },
            ).scalar_one()

            db.execute(
                text(
                    "INSERT INTO chunks (document_id, parent_id, content, embedding) "
                    "VALUES (:document_id, :parent_id, :content, CAST(:embedding AS vector))"
                ),
                [
                    {
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "content": c.content,
                        "embedding": str(c.embedding),
                    }
                    for c in parent.children
                ],
            )
        db.commit()
        return document_id
    except Exception:
        db.rollback()
        raise


def save_flat_document(
    db: Session,
    source: str,
    title: str,
    parents: list[ParentChunk],
    content_hash: str | None = None,
) -> int:
    """Lưu tài liệu theo schema flat """
    try:
        document_id = _replace_document(db, source, title, content_hash)
        rows = [
            {"document_id": document_id, "content": c.content, "embedding": str(c.embedding)}
            for parent in parents
            for c in parent.children
        ]
        if rows:
            db.execute(
                text(
                    "INSERT INTO chunks (document_id, content, embedding) "
                    "VALUES (:document_id, :content, CAST(:embedding AS vector))"
                ),
                rows,
            )
        db.commit()
        return document_id
    except Exception:
        db.rollback()
        raise
