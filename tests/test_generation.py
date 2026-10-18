"""Test LLM bằng client giả, không gọi mạng và không cần API key thật."""

from types import SimpleNamespace

import pytest

from src.generation import get_llm


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_llm("khong_co")


def test_missing_api_key_raises(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    for provider in ("openai", "gemini"):
        with pytest.raises(ValueError):
            get_llm(provider)


def test_openai_generate_sends_system_and_prompt(monkeypatch):
    from src.generation.openai_llm import OpenAILLM

    calls = {}

    def create(model, messages):
        calls.update(model=model, messages=messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="xin chào"))])

    llm = OpenAILLM(model="m", api_key="k")
    monkeypatch.setattr(llm.client.chat.completions, "create", create)

    assert llm.generate("câu hỏi", system="trả lời ngắn") == "xin chào"
    assert calls["model"] == "m"
    assert calls["messages"] == [
        {"role": "system", "content": "trả lời ngắn"},
        {"role": "user", "content": "câu hỏi"},
    ]


def test_gemini_generate_sends_system_and_prompt(monkeypatch):
    from src.generation.gemini_llm import GeminiLLM

    calls = {}

    def generate_content(model, contents, config):
        calls.update(model=model, contents=contents, system=config.system_instruction)
        return SimpleNamespace(text="xin chào")

    llm = GeminiLLM(model="m", api_key="k")
    monkeypatch.setattr(llm.client.models, "generate_content", generate_content)

    assert llm.generate("câu hỏi", system="trả lời ngắn") == "xin chào"
    assert calls == {"model": "m", "contents": "câu hỏi", "system": "trả lời ngắn"}


def test_factory_picks_provider_from_config(monkeypatch):
    from src.config import settings
    from src.generation.gemini_llm import GeminiLLM
    from src.generation.openai_llm import OpenAILLM

    monkeypatch.setattr(settings, "openai_api_key", "k")
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings.llm, "provider", "openai")
    assert isinstance(get_llm(), OpenAILLM)
    monkeypatch.setattr(settings.llm, "provider", "gemini")
    assert isinstance(get_llm(), GeminiLLM)
