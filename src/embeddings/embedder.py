from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.chunking import ParentChunk
from src.config import settings


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding.model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()


def embed_parents(parents: list[ParentChunk]) -> None:
    children = [child for parent in parents for child in parent.children]
    vectors = embed_texts([child.content for child in children])
    for child, vector in zip(children, vectors):
        child.embedding = vector
