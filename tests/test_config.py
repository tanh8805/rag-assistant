from src.config import settings


def test_each_strategy_has_its_own_database():
    assert settings.database_url_for("parent_child").endswith("/rag_parent_child")
    assert settings.database_url_for("flat").endswith("/rag_flat")


def test_default_database_follows_strategy_in_config():
    assert settings.database_url == settings.database_url_for(settings.chunking.strategy)
