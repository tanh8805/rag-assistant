from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class BaseLoader(ABC):
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        pass