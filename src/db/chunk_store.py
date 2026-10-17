import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.chunking import ParentChunk


def save_document(db: Session, source: str, title: str, parents: list[ParentChunk]) -> int:
    try:
        db.execute(text("DELETE FROM documents WHERE source = :source"), {"source": source})
        document_id = db.execute(
            text("INSERT INTO documents (source, title) VALUES (:source, :title) RETURNING id"),
            {"source": source, "title": title},
        ).scalar_one()

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
                    "INSERT INTO chunks (parent_id, content, embedding) "
                    "VALUES (:parent_id, :content, CAST(:embedding AS vector))"
                ),
                [
                    {"parent_id": parent_id, "content": c.content, "embedding": str(c.embedding)}
                    for c in parent.children
                ],
            )
        db.commit()
        return document_id
    except Exception:
        db.rollback()
        raise
