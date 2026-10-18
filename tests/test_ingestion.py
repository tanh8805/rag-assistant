import pymupdf
import pytest

from src.ingestion import get_loader, load_file
from src.ingestion.hashing import hash_file
from src.ingestion.text_loader import TextLoader


def test_load_markdown_and_text(tmp_path):
    for name in ("a.md", "b.txt"):
        path = tmp_path / name
        path.write_text("Xin chào các bạn", encoding="utf-8")
        docs = load_file(path)
        assert len(docs) == 1
        assert docs[0].content == "Xin chào các bạn"
        assert docs[0].source == str(path)
        assert docs[0].metadata["title"] == path.stem


def test_load_html_removes_script_and_gets_title(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Tiêu đề</title><style>p{}</style></head>"
        "<body><script>alert(1)</script><p>Nội dung</p></body></html>",
        encoding="utf-8",
    )
    doc = load_file(path)[0]
    assert doc.metadata["title"] == "Tiêu đề"
    assert "Nội dung" in doc.content
    assert "alert" not in doc.content


def test_load_pdf_one_document_per_non_empty_page(tmp_path):
    path = tmp_path / "file.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Page one")
    pdf.new_page()  # trang trống, phải bị bỏ qua
    pdf.new_page().insert_text((72, 72), "Page three")
    pdf.save(path)
    pdf.close()

    docs = load_file(path)
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert "Page one" in docs[0].content


def test_loader_chosen_by_extension_case_insensitive(tmp_path):
    assert isinstance(get_loader(tmp_path / "A.MD"), TextLoader)


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        get_loader("file.docx")


def test_hash_file_depends_on_content_not_name(tmp_path):
    a, b, c = tmp_path / "a.txt", tmp_path / "b.md", tmp_path / "c.txt"
    a.write_text("cùng nội dung", encoding="utf-8")
    b.write_text("cùng nội dung", encoding="utf-8")
    c.write_text("nội dung khác", encoding="utf-8")
    assert hash_file(a) == hash_file(b)
    assert hash_file(a) != hash_file(c)
