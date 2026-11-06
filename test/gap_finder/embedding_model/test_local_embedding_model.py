from unittest.mock import Mock

import numpy as np
import pytest

from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel


def test_encode_forwards_batch_options_and_returns_float32():
    sentence_transformer = Mock()
    sentence_transformer.encode.return_value = [[1.0, 2.0], [3.0, 4.0]]
    model = LocalEmbeddingModel(sentence_transformer)
    embeddings = model.encode(
        ["first", "second"],
        show_progress_bar=True,
        batch_size=2,
    )

    sentence_transformer.encode.assert_called_once_with(
        ["first", "second"],
        show_progress_bar=True,
        batch_size=2,
    )
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        embeddings,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_embed_maps_bertopic_verbose_to_progress_bar():
    model = LocalEmbeddingModel(Mock())
    model.encode = Mock(return_value=np.array([[1.0]], dtype=np.float32))

    result = model.embed(["document"], verbose=True)

    model.encode.assert_called_once_with(
        ["document"],
        show_progress_bar=True,
        batch_size=32,
    )
    np.testing.assert_array_equal(result, np.array([[1.0]], dtype=np.float32))


def test_encode_rejects_invalid_batch_size():
    model = LocalEmbeddingModel(Mock())

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        model.encode(["document"], batch_size=0)


def test_lifecycle_does_not_move_or_release_caller_owned_model():
    sentence_transformer = Mock()
    sentence_transformer.device = "cuda"
    model = LocalEmbeddingModel(sentence_transformer)
    sentence_transformer.to.reset_mock()

    model.activate()
    model.deactivate()
    model.cleanup()

    assert model.client is sentence_transformer
    sentence_transformer.to.assert_not_called()


def test_owned_model_is_deactivated_and_released(monkeypatch):
    sentence_transformer = Mock()
    sentence_transformer.device = "cuda"
    constructor = Mock(return_value=sentence_transformer)
    monkeypatch.setattr(
        "gap_finder.embedding_model.local_embedding_model.SentenceTransformer",
        constructor,
    )
    monkeypatch.setattr("torch.cuda.is_available", Mock(return_value=False))
    model = LocalEmbeddingModel("local-model")

    model.activate()
    model.deactivate()

    assert model.client is sentence_transformer
    sentence_transformer.to.assert_called_once()

    model.cleanup()

    assert model.client is None
