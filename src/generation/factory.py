from src.config import settings

from .base import BaseLLM


def get_llm(provider: str | None = None) -> BaseLLM:
    provider = provider or settings.llm.provider
    if provider == "gemini":
        from .gemini_llm import GeminiLLM

        return GeminiLLM()
    if provider == "openai":
        from .openai_llm import OpenAILLM

        return OpenAILLM()
    raise ValueError(f"Không hỗ trợ llm provider '{provider}'. Hỗ trợ: gemini, openai")
