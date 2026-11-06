"""
Unit tests for FaissVectorStore / NumpyVectorStore.

Both implementations of VectorStore are exercised through the same
parametrized tests (`vector_store_cls` fixture from conftest.py) to enforce
that they behave identically. These tests use synthetic embeddings only, so
they run fully offline and require no model or PDF runtime.
"""
import numpy as np
import pytest


def make_embeddings(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float64).astype(np.float32)


class TestAddAndSearch:
    def test_search_retrieves_exact_match_as_top_hit(self, vector_store_cls):
        embeddings = make_embeddings(5, 8)
        chunks = [f"chunk-{i}" for i in range(5)]
        store = vector_store_cls(dim=8)
        store.add(embeddings, chunks)

        query = embeddings[2:3].copy()
        D, I, vecs, retrieved_chunks = store.search(query, k=1)

        assert I[0][0] == 2
        assert retrieved_chunks[0] == "chunk-2"
        assert D[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_search_accepts_1d_query(self, vector_store_cls):
        embeddings = make_embeddings(3, 4)
        store = vector_store_cls(dim=4)
        store.add(embeddings, ["a", "b", "c"])

        query_1d = embeddings[1].copy()
        D, I, vecs, chunks = store.search(query_1d, k=1)

        assert I[0][0] == 1
        assert chunks[0] == "b"

    def test_get_all_returns_everything_added(self, vector_store_cls):
        embeddings = make_embeddings(4, 6)
        chunks = ["a", "b", "c", "d"]
        store = vector_store_cls(dim=6)
        store.add(embeddings, chunks)

        all_vectors, all_chunks = store.get_all()

        assert all_vectors.shape[0] == 4
        assert all_chunks == chunks

    def test_add_across_multiple_calls_accumulates(self, vector_store_cls):
        store = vector_store_cls(dim=4)
        store.add(make_embeddings(2, 4, seed=1), ["a", "b"])
        store.add(make_embeddings(3, 4, seed=2), ["c", "d", "e"])

        vectors, chunks = store.get_all()
        assert vectors.shape[0] == 5
        assert chunks == ["a", "b", "c", "d", "e"]


class TestEmptyStore:
    def test_search_on_empty_store_does_not_crash(self, vector_store_cls):
        """Regression test: searching an empty FaissVectorStore used to
        segfault the process (faiss.Index.reconstruct(-1) on an index with
        ntotal == 0 is undefined behaviour in the C++ layer), and searching an
        empty NumpyVectorStore raised IndexError (negative-indexing a
        zero-length array). Both must now return gracefully.
        """
        store = vector_store_cls(dim=4)
        query = make_embeddings(1, 4)

        D, I, vecs, chunks = store.search(query, k=3)

        assert D.shape == (1, 3)
        assert I.shape == (1, 3)
        assert (I == -1).all()
        assert vecs == []
        assert chunks == []

    def test_get_all_on_empty_store(self, vector_store_cls):
        store = vector_store_cls(dim=4)
        vectors, chunks = store.get_all()

        assert vectors.shape == (0, 4)
        assert chunks == []


class TestSearchPastNtotal:
    def test_k_greater_than_ntotal_pads_with_sentinel(self, vector_store_cls):
        """
        Documents current behaviour rather than desired behaviour: when
        k > ntotal, missing neighbours are padded with index -1 / a large
        negative score. Because -1 is also a valid Python "last element"
        index, the padding slots come back as a duplicate of the last real
        hit's chunk instead of being omitted - this is pre-existing,
        surprising behaviour that callers must be aware of.
        """
        embeddings = make_embeddings(2, 4)
        chunks = ["a", "b"]
        store = vector_store_cls(dim=4)
        store.add(embeddings, chunks)

        k = 5
        D, I, vecs, retrieved_chunks = store.search(embeddings[0:1].copy(), k=k)

        valid = I[0] != -1
        assert valid.sum() == 2
        assert list(I[0][2:]) == [-1, -1, -1]
        assert retrieved_chunks[2:] == ["b"] * 3


class TestNoMutationOfCallerArrays:
    """Both stores normalize embeddings/queries internally; neither should
    mutate the caller's arrays as a side effect (FaissVectorStore used to,
    via in-place faiss.normalize_L2)."""

    def test_add_does_not_mutate_caller_embeddings(self, vector_store_cls):
        embeddings = make_embeddings(3, 4)
        original = embeddings.copy()
        store = vector_store_cls(dim=4)

        store.add(embeddings, ["a", "b", "c"])

        assert np.array_equal(embeddings, original)

    def test_search_does_not_mutate_caller_query(self, vector_store_cls):
        embeddings = make_embeddings(3, 4)
        store = vector_store_cls(dim=4)
        store.add(embeddings, ["a", "b", "c"])

        query = embeddings[0].copy()
        original_query = query.copy()
        store.search(query, k=1)

        assert np.array_equal(query, original_query)


class TestPersistence:
    def test_save_and_load_round_trip(self, vector_store_cls, tmp_path):
        embeddings = make_embeddings(4, 5)
        chunks = ["a", "b", "c", "d"]
        store = vector_store_cls(dim=5)
        store.add(embeddings, chunks)

        store.save_to(tmp_path, file_name="my_store")
        assert (tmp_path / "my_store.json").is_file()
        assert not (tmp_path / "my_store.pkl").exists()
        loaded = vector_store_cls.load_from(tmp_path, file_name="my_store")

        orig_vectors, orig_chunks = store.get_all()
        loaded_vectors, loaded_chunks = loaded.get_all()

        assert loaded_chunks == orig_chunks
        assert np.allclose(loaded_vectors, orig_vectors)

        D, I, _, retrieved_chunks = loaded.search(embeddings[1:2].copy(), k=1)
        assert retrieved_chunks[0] == "b"

    def test_save_pdf_vector_store_and_load_round_trip(self, vector_store_cls, tmp_path):
        embeddings = make_embeddings(2, 3)
        chunks = ["x", "y"]
        store = vector_store_cls(dim=3)
        store.add(embeddings, chunks)

        store.save_pdf_vector_store(tmp_path, "sentence-transformers/all-MiniLM-L6-v2")
        loaded = vector_store_cls.load_pdf_vector_store(tmp_path, "sentence-transformers/all-MiniLM-L6-v2")

        assert loaded.get_all()[1] == chunks
