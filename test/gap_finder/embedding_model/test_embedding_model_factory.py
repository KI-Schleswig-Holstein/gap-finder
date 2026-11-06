import subprocess
import sys
from unittest.mock import Mock

import pytest
from sentence_transformers import SentenceTransformer

from gap_finder.embedding_model import factory
from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel
from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel


def test_string_creates_owned_sentence_transformer(monkeypatch):
    client = Mock(spec=SentenceTransformer)
    client.device = "cuda"
    client.encode.return_value = [[1.0]]
    constructor = Mock(return_value=client)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    monkeypatch.setattr(
        "gap_finder.embedding_model.local_embedding_model.SentenceTransformer",
        constructor,
    )

    result = factory.create_embedding_model("local-model")

    assert isinstance(result, LocalEmbeddingModel)
    assert result.client is None
    constructor.assert_not_called()

    result.encode(["text"])

    assert result.client is client
    constructor.assert_called_once_with("local-model")


def test_reuses_sentence_transformer():
    client = Mock(spec=SentenceTransformer)
    client.device = "cpu"

    result = factory.create_embedding_model(client)

    assert isinstance(result, LocalEmbeddingModel)
    assert result.client is client


def test_reuses_openai_embedding_client():
    configured_client = OpenAIEmbeddingClient(client=Mock(), model_name="api-model")

    result = factory.create_embedding_model(configured_client)

    assert isinstance(result, OpenAIEmbeddingModel)
    assert result.client is configured_client.client
    assert result.model_name == "api-model"


def test_rejects_unsupported_model():
    with pytest.raises(TypeError, match="embedding_model"):
        factory.create_embedding_model(object())


def test_returns_existing_embedding_model_unchanged():
    embedding_model = Mock(spec=EmbeddingModel)

    result = factory.create_embedding_model(embedding_model)

    assert result is embedding_model


def test_api_model_does_not_import_local_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from unittest.mock import Mock
from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.factory import create_embedding_model

create_embedding_model(OpenAIEmbeddingClient(model_name="api-model", client=Mock()))
assert "transformers" not in sys.modules
assert "sentence_transformers" not in sys.modules
assert "bertopic" not in sys.modules
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
