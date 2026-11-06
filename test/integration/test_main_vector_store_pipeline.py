"""GapFinder construction tests using real PDFs and embedding models."""

import numpy as np
import pytest

from gap_finder.main import GapFinder
from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

pytestmark = pytest.mark.vs


@pytest.mark.parametrize("vector_store_cls", [NumpyVectorStore, FaissVectorStore])
def test_from_chunks_builds_searchable_vector_store(
    vector_store_cls,
    embedding_model_name,
    document_chunks,
    mock_language_model,
):
    gap_finder = GapFinder.from_chunks(
        chunks=document_chunks,
        embedding_model=embedding_model_name,
        language_model=mock_language_model,
        vector_store_cls=vector_store_cls,
    )

    assert isinstance(gap_finder.vector_store, vector_store_cls)
    vectors, chunks = gap_finder.vector_store.get_all()
    assert vectors.shape[0] == len(document_chunks)
    assert chunks == document_chunks


def test_from_chunks_defaults_to_numpy_vector_store(
    embedding_model_name, mock_language_model
):
    gap_finder = GapFinder.from_chunks(
        chunks=["hello world", "goodbye world"],
        embedding_model=embedding_model_name,
        language_model=mock_language_model,
    )

    assert isinstance(gap_finder.vector_store, NumpyVectorStore)


@pytest.mark.parametrize("vector_store_cls", [NumpyVectorStore, FaissVectorStore])
def test_from_pdf_builds_searchable_vector_store(
    vector_store_cls,
    embedding_model,
    embedding_model_name,
    pdf_dir,
    mock_language_model,
):
    gap_finder = GapFinder.from_pdf(
        embedding_model=embedding_model_name,
        language_model=mock_language_model,
        folder_path=str(pdf_dir),
        chunk_size=1000,
        chunk_overlap=100,
        vector_store_cls=vector_store_cls,
    )

    assert isinstance(gap_finder.vector_store, vector_store_cls)
    vectors, chunks = gap_finder.vector_store.get_all()
    assert vectors.shape[0] == len(chunks)
    assert vectors.shape[0] > 0

    query_text = chunks[0]
    query_embedding = embedding_model.encode([query_text]).astype(np.float32)
    _, _, _, retrieved_chunks = gap_finder.vector_store.search(query_embedding, k=1)

    assert retrieved_chunks[0] == query_text
