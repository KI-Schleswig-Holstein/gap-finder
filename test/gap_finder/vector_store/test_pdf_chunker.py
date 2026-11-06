from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gap_finder.vector_store import pdf_chunker
from gap_finder.vector_store.pdf_chunker import PDFChunker


@pytest.fixture
def text_splitter(monkeypatch):
    splitter = Mock()
    splitter.split_text.side_effect = lambda text: [text]
    factory = Mock(return_value=splitter)
    monkeypatch.setattr(pdf_chunker, "_create_text_splitter", factory)
    return splitter, factory


def test_chunk_pdf_partitions_pdf_with_fast_strategy_and_languages(
    monkeypatch,
    text_splitter,
):
    calls = []

    def partition_pdf(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(text="first page"), SimpleNamespace(text="second page")]

    monkeypatch.setattr(pdf_chunker, "partition_pdf", partition_pdf)

    chunks = PDFChunker(chunk_size=1000, chunk_overlap=0, languages=["eng"]).chunk_pdf(Path("paper.pdf"))

    assert calls == [{"filename": "paper.pdf", "strategy": "fast", "languages": ["eng"]}]
    assert chunks == ["first page\n\nsecond page"]


def test_chunk_pdf_ignores_empty_element_text(monkeypatch, text_splitter):
    monkeypatch.setattr(
        pdf_chunker,
        "partition_pdf",
        lambda **_: [
            SimpleNamespace(text="first"),
            SimpleNamespace(text=""),
            SimpleNamespace(text=None),
            SimpleNamespace(text="second"),
        ],
    )

    chunks = PDFChunker(chunk_size=1000, chunk_overlap=0).chunk_pdf(Path("paper.pdf"))

    assert chunks == ["first\n\nsecond"]


def test_default_languages_are_english():
    assert PDFChunker().languages == ["eng"]


def test_custom_languages_are_preserved():
    languages = ["eng", "deu"]

    assert PDFChunker(languages=languages).languages is languages


def test_empty_languages_are_preserved():
    assert PDFChunker(languages=[]).languages == []


def test_chunk_pdf_configures_and_uses_text_splitter(monkeypatch, text_splitter):
    monkeypatch.setattr(
        pdf_chunker,
        "partition_pdf",
        lambda **_: [SimpleNamespace(text="alpha beta gamma delta")],
    )

    splitter, factory = text_splitter
    splitter.split_text.side_effect = None
    splitter.split_text.return_value = ["alpha", "beta"]

    chunks = PDFChunker(chunk_size=10, chunk_overlap=2).chunk_pdf(Path("paper.pdf"))

    factory.assert_called_once_with(chunk_size=10, chunk_overlap=2)
    splitter.split_text.assert_called_once_with("alpha beta gamma delta")
    assert chunks == ["alpha", "beta"]
