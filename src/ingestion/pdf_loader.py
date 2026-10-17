from pathlib import Path

import pymupdf

from .base import BaseLoader, Document


class PDFLoader(BaseLoader):
    extensions = (".pdf",)

    def load(self, path: Path) -> list[Document]:
        docs = []
        with pymupdf.open(path) as pdf:
            for index in range(pdf.page_count):
                page_no = index + 1
                text = str(pdf.load_page(index).get_text("text")).strip()
                if text:
                    docs.append(
                        Document(
                            content=text,
                            source=str(path),
                            metadata={"title": path.stem, "page": page_no},
                        )
                    )
        return docs
