from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Gửi prompt, trả về câu trả lời dạng text."""
