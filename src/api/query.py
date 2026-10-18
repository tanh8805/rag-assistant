from functools import lru_cache

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_db
from src.generation import SYSTEM_PROMPT, BaseLLM, build_prompt, get_llm
from src.retrieval import get_parents, get_retriever

router = APIRouter()


@lru_cache
def default_llm() -> BaseLLM:
    return get_llm()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    retriever: str = Field("rrf", pattern="^(dense|bm25|rrf)$")
    top_k: int = Field(settings.retrieval.top_k, ge=1, le=100)
    document_ids: list[int] | None = None 


class Source(BaseModel):
    source: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db), llm: BaseLLM = Depends(default_llm)):
    
    chunks = get_retriever(req.retriever, db, req.document_ids).retrieve(req.question, req.top_k)
    
    parents = get_parents(db, chunks)[: settings.rerank.top_n] if chunks else []
    if not parents:
        return QueryResponse(answer="Không tìm thấy thông tin liên quan trong tài liệu.", sources=[])

    answer = llm.generate(build_prompt(req.question, parents), system=SYSTEM_PROMPT)
    return QueryResponse(answer=answer, sources=[Source(source=p.source, content=p.content) for p in parents])
