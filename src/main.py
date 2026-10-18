from fastapi import FastAPI

from src.api import ingest, query

app = FastAPI(title="RAG Assistant")
app.include_router(ingest.router)
app.include_router(query.router)
