"""
Integration smoke test for PDFChunker against a real PDF.

PDF parsing via `unstructured` is comparatively slow, so this is guarded by
the `vs` marker like the other heavier vector-store pipeline tests.
"""

import pytest

from gap_finder.vector_store.pdf_chunker import PDFChunker

pytestmark = pytest.mark.vs


def test_chunk_pdf_returns_nonempty_text_chunks(pdf_dir):
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk_pdf(pdf_dir / "synthetic_language_systems.pdf")

    assert chunks
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks)
