from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel
from gap_finder.embedding_model.client import OpenAIEmbeddingClient


def embedding(index: int, values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(index=index, embedding=values)


def test_encode_batches_requests_preserves_order_and_returns_float32():
    client = Mock()
    client.embeddings.create.side_effect = [
        SimpleNamespace(
            data=[embedding(1, [2.0, 20.0]), embedding(0, [1.0, 10.0])]
        ),
        SimpleNamespace(
            data=[embedding(1, [4.0, 40.0]), embedding(0, [3.0, 30.0])]
        ),
        SimpleNamespace(data=[embedding(0, [5.0, 50.0])]),
    ]
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    embeddings = model.encode(
        ["one", "two", "three", "four", "five"],
        batch_size=2,
    )

    assert client.embeddings.create.call_args_list == [
        (((), {"model": "api-model", "input": ["one", "two"]})),
        (((), {"model": "api-model", "input": ["three", "four"]})),
        (((), {"model": "api-model", "input": ["five"]})),
    ]
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        embeddings,
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize(
    ("text_count", "batch_size", "expected_calls"),
    [(1, 2, 1), (2, 2, 1), (3, 2, 2)],
)
def test_encode_handles_batch_boundaries(text_count, batch_size, expected_calls):
    client = Mock()

    def create(*, model, input):
        return SimpleNamespace(
            data=[embedding(index, [float(index)]) for index, _ in enumerate(input)]
        )

    client.embeddings.create.side_effect = create
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    result = model.encode([str(index) for index in range(text_count)], batch_size=batch_size)

    assert client.embeddings.create.call_count == expected_calls
    assert result.shape == (text_count, 1)


def test_encode_enables_progress_bar(monkeypatch):
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[embedding(0, [1.0])]
    )
    progress = Mock(side_effect=lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        "gap_finder.embedding_model.openai_embedding_model.tqdm",
        progress,
    )
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    model.encode(["document"], show_progress_bar=True)

    progress.assert_called_once()
    assert progress.call_args.kwargs == {"disable": False}


def test_encode_empty_input_avoids_api_call():
    client = Mock()
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    embeddings = model.encode([])

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32
    client.embeddings.create.assert_not_called()


def test_encode_rejects_invalid_batch_size():
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=Mock())
    )

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        model.encode(["document"], batch_size=0)


def test_encode_propagates_api_failure():
    client = Mock()
    client.embeddings.create.side_effect = RuntimeError("request failed")
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    with pytest.raises(RuntimeError, match="request failed"):
        model.encode(["document"])


def test_cleanup_is_a_no_op_for_api_model():
    client = Mock()
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingClient(model_name="api-model", client=client)
    )

    model.cleanup()

    assert model.client is client
