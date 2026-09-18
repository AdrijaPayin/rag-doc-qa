import pytest

from app.document_loader import chunk_text, read_txt, load_document, load_and_chunk_documents


def test_chunk_text_basic_splitting():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[-1]) == 300


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap_creates_shared_content():
    text = "0123456789" * 60
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert chunks[0][-20:] == chunks[1][:20]


def test_read_txt_reads_file_contents(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world", encoding="utf-8")
    assert read_txt(str(file_path)) == "Hello world"


def test_load_document_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(str(file_path))


def test_load_and_chunk_documents_reads_all_txt_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("Some content here.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("More content here.", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("Should be ignored.", encoding="utf-8")

    chunks = load_and_chunk_documents(str(tmp_path))
    sources = {c["source"] for c in chunks}

    assert sources == {"doc1.txt", "doc2.txt"}


def test_load_and_chunk_documents_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        load_and_chunk_documents("this/folder/does/not/exist")
