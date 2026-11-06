"""Vector-store integration tests using a real PDF and embedding model."""

import numpy as np
import pytest

from gap_finder.vector_store.faiss_vector_store import FaissVectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

pytestmark = pytest.mark.vs


def test_retrieves_source_chunk_as_top_hit(embedding_model, populated_store):
    store, chunks = populated_store
    vectors, _ = store.get_all()
    assert vectors.shape[0] == len(chunks)

    query_idx = len(chunks) // 2
    query_embedding = embedding_model.encode([chunks[query_idx]]).astype(np.float32)

    scores, indices, _, retrieved_chunks = store.search(query_embedding, k=1)

    assert indices[0][0] == query_idx
    assert retrieved_chunks[0] == chunks[query_idx]
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


def test_faiss_and_numpy_agree_on_top_hits(
    embedding_model,
    document_chunks,
    document_embeddings,
):
    faiss_store = FaissVectorStore(dim=document_embeddings.shape[1])
    faiss_store.add(document_embeddings.copy(), list(document_chunks))

    numpy_store = NumpyVectorStore(dim=document_embeddings.shape[1])
    numpy_store.add(document_embeddings.copy(), list(document_chunks))

    query_embedding = embedding_model.encode([document_chunks[3]]).astype(np.float32)

    f_scores, f_idx, _, f_chunks = faiss_store.search(query_embedding.copy(), k=5)
    n_scores, n_idx, _, n_chunks = numpy_store.search(query_embedding.copy(), k=5)

    assert list(f_idx[0]) == list(n_idx[0])
    assert f_chunks == n_chunks
    assert np.allclose(f_scores, n_scores, atol=1e-4)
