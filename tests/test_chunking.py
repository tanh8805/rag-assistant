import pytest

from src.chunking import RecursiveChunker, get_chunker
from src.ingestion import Document

TEXT = "\n\n".join(f"Đây là câu số {i}. Nó nói về chủ đề số {i} của tài liệu." for i in range(40))


def make_doc() -> Document:
    return Document(content=TEXT, source="doc.txt", metadata={"title": "doc"})


def make_chunker() -> RecursiveChunker:
    return RecursiveChunker(parent_size=300, parent_overlap=20, child_size=100, child_overlap=10)


def test_sizes_do_not_exceed_limit():
    parents = make_chunker().split(make_doc())
    assert len(parents) > 1
    for parent in parents:
        assert len(parent.content) <= 300
        for child in parent.children:
            assert len(child.content) <= 100


def test_children_come_from_their_parent():
    for parent in make_chunker().split(make_doc()):
        assert parent.children
        for child in parent.children:
            assert child.content in parent.content


def test_parent_index_points_to_parent():
    parents = make_chunker().split(make_doc())
    for index, parent in enumerate(parents):
        assert all(child.parent_index == index for child in parent.children)


def test_source_and_metadata_are_kept():
    doc = make_doc()
    parents = make_chunker().split(doc)
    assert all(p.source == "doc.txt" for p in parents)
    assert all(p.metadata == {"title": "doc"} for p in parents)
    assert parents[0].metadata is not doc.metadata  


def test_short_text_gives_one_parent_one_child():
    parents = make_chunker().split(Document("Ngắn thôi", "a.txt"))
    assert len(parents) == 1
    assert [c.content for c in parents[0].children] == ["Ngắn thôi"]


def test_factory_uses_settings():
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.parent_size == 3000
    assert chunker.child_size == 600


def test_factory_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_chunker("khong-co")
