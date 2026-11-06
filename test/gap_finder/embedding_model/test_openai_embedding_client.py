from unittest.mock import Mock

import pytest

from gap_finder import MissingAPIKeyError
from gap_finder.embedding_model.client import OpenAIEmbeddingClient


def test_uses_api_key_from_environment(monkeypatch):
    openai = Mock()
    monkeypatch.setattr("gap_finder.embedding_model.client.OpenAI", openai)
    monkeypatch.setenv("GAP_FINDER_API_KEY", "environment-key")

    client = OpenAIEmbeddingClient(
        model_name="embedding-model",
        base_url="http://example.test/v1",
    )

    openai.assert_called_once_with(
        base_url="http://example.test/v1",
        api_key="environment-key",
    )
    assert client.client is openai.return_value


def test_explicit_api_key_takes_precedence(monkeypatch):
    openai = Mock()
    monkeypatch.setattr("gap_finder.embedding_model.client.OpenAI", openai)
    monkeypatch.setenv("GAP_FINDER_API_KEY", "environment-key")

    OpenAIEmbeddingClient(model_name="embedding-model", api_key="explicit-key")

    openai.assert_called_once_with(base_url=None, api_key="explicit-key")


def test_reuses_preconfigured_client_without_resolving_api_key(monkeypatch):
    monkeypatch.delenv("GAP_FINDER_API_KEY", raising=False)
    configured_client = Mock()

    client = OpenAIEmbeddingClient(
        model_name="embedding-model",
        client=configured_client,
    )

    assert client.client is configured_client


def test_does_not_use_openai_api_key(monkeypatch):
    monkeypatch.delenv("GAP_FINDER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")

    with pytest.raises(MissingAPIKeyError, match="GAP_FINDER_API_KEY"):
        OpenAIEmbeddingClient(model_name="embedding-model")
