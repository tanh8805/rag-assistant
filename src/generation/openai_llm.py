from langsmith import traceable
from openai import OpenAI

from src.config import settings

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("Thiếu OPENAI_API_KEY trong .env")
        self.model = model or settings.llm.openai_model
        self.client = OpenAI(api_key=api_key)

    @traceable(run_type="llm", name="openai_generate")
    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""
