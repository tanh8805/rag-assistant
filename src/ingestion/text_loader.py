from pathlib import Path

from .base import BaseLoader, Document


class TextLoader(BaseLoader):
    extensions = (".md", ".txt")

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8")
        return [Document(content=content, source=str(path), metadata={"title": path.stem})]
