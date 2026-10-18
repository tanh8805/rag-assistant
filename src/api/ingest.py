from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import ROOT
from src.db import get_db
from src.ingestion import get_loader
from src.ingestion.pipeline import ingest_file

UPLOAD_DIR = ROOT / "data" / "uploads"

router = APIRouter()


class IngestResponse(BaseModel):
    filename: str
    status: str  # "ingested" | "duplicate"
    document_id: int | None


@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "").name
    try:
        get_loader(filename)  
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    existed = path.exists()
    path.write_bytes(file.file.read())

    document_id = ingest_file(db, path)
    if document_id is None:
        if not existed:
            path.unlink()  
        return IngestResponse(filename=filename, status="duplicate", document_id=None)
    return IngestResponse(filename=filename, status="ingested", document_id=document_id)
