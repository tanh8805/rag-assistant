from functools import lru_cache

import torch
from langsmith import traceable
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval import RetrievedParent


@lru_cache
def get_reranker() -> CrossEncoder:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # fp16 để vừa VRAM cùng model embedding
    return CrossEncoder(settings.rerank.model, max_length=768, model_kwargs={"dtype": dtype})


@traceable(run_type="chain", name="rerank")
def rerank(query: str, parents: list[RetrievedParent], top_n: int = settings.rerank.top_n) -> list[RetrievedParent]:
    """Chấm điểm từng cặp (câu hỏi, văn bản) bằng cross-encoder, trả về top_n văn bản điểm cao nhất."""
    if not parents:
        return []
    scores = get_reranker().predict([(query, p.content) for p in parents], batch_size=4)
    ranked = sorted(zip(scores, parents), key=lambda pair: pair[0], reverse=True)
    return [p for _, p in ranked[:top_n]]
