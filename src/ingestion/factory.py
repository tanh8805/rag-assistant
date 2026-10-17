from pathlib import Path

from .base import BaseLoader, Document
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader

LOADERS: dict[str, BaseLoader] = {
    ".md": TextLoader(),
    ".txt": TextLoader(),
    ".pdf": PDFLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


def get_loader(path: str | Path) -> BaseLoader:
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Không hỗ trợ định dạng '{ext}'. Hỗ trợ: {', '.join(LOADERS)}")
    return LOADERS[ext]


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    return get_loader(path).load(path)
