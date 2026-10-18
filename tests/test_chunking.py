import pytest

from src.chunking import FlatChunker, ParentChildChunker, get_chunker
from src.ingestion import Document

TEXT = "\n\n".join(f"Đây là câu số {i}. Nó nói về chủ đề số {i} của tài liệu." for i in range(40))


def make_doc() -> Document:
    return Document(content=TEXT, source="doc.txt", metadata={"title": "doc"})


def make_chunker() -> ParentChildChunker:
    return ParentChildChunker(parent_size=300, child_size=100, child_overlap=10)


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


def test_parent_child_parents_do_not_overlap():
    parents = make_chunker().split(make_doc())
    assert "".join(p.content.replace("\n", "") for p in parents) == TEXT.replace("\n", "")


def make_flat_chunker() -> FlatChunker:
    return FlatChunker(chunk_size=100, chunk_overlap=20)


def test_flat_chunk_is_parent_with_one_identical_child():
    parents = make_flat_chunker().split(make_doc())
    assert len(parents) > 1
    for index, parent in enumerate(parents):
        assert len(parent.content) <= 100
        assert [c.content for c in parent.children] == [parent.content]
        assert parent.children[0].parent_index == index
    assert all(p.source == "doc.txt" and p.metadata == {"title": "doc"} for p in parents)


def test_flat_chunks_overlap():
    chunker = FlatChunker(chunk_size=50, chunk_overlap=20)
    parents = chunker.split(Document(" ".join(f"tu{i}" for i in range(60)), "a.txt"))
    for before, after in zip(parents, parents[1:]):
        assert set(before.content.split()) & set(after.content.split())  # có từ chung


def test_factory_uses_settings():
    chunker = get_chunker()
    assert isinstance(chunker, ParentChildChunker)
    assert (chunker.parent_size, chunker.child_size, chunker.child_overlap) == (3000, 600, 80)

    flat = get_chunker("flat")
    assert isinstance(flat, FlatChunker)
    assert (flat.chunk_size, flat.chunk_overlap) == (1000, 100)


def test_factory_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_chunker("khong-co")
