from pathlib import Path

from sqlalchemy.orm import Session

from src.chunking import get_chunker
from src.config import ChunkingStrategy, settings
from src.db import document_exists, save_document, save_flat_document
from src.embeddings import embed_parents

from .factory import load_file
from .hashing import hash_file


def ingest_file(db: Session, path: str | Path, strategy: ChunkingStrategy | None = None) -> int | None:
    strategy = strategy or settings.chunking.strategy
    content_hash = hash_file(path)
    if document_exists(db, content_hash):
        return None

    docs = load_file(path)
    chunker = get_chunker(strategy)
    parents = [p for doc in docs for p in chunker.split(doc)]
    embed_parents(parents)

    save = save_flat_document if strategy == "flat" else save_document
    title = docs[0].metadata.get("title", Path(path).stem) if docs else Path(path).stem
    return save(db, str(path), title, parents, content_hash)
