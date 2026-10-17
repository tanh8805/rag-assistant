from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseLoader, Document


class HTMLLoader(BaseLoader):
    extensions = (".html", ".htm")

    def load(self, path: Path) -> list[Document]:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
        text = soup.get_text(separator="\n", strip=True)
        return [Document(content=text, source=str(path), metadata={"title": title})]
