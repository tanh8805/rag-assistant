from google import genai
from google.genai import types

from src.config import settings

from .base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("Thiếu GEMINI_API_KEY trong .env")
        self.model = model or settings.llm.gemini_model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""
