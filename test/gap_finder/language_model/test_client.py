from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gap_finder import MissingAPIKeyError
from gap_finder.language_model.client import ClientError, OpenAIClient


def test_uses_api_key_from_environment(monkeypatch):
    openai = Mock()
    monkeypatch.setattr("gap_finder.language_model.client.OpenAI", openai)
    monkeypatch.setenv("GAP_FINDER_API_KEY", "environment-key")

    OpenAIClient(model_name="test-model", base_url="http://example.test/v1")

    openai.assert_called_once_with(
        base_url="http://example.test/v1",
        api_key="environment-key",
    )


def test_explicit_api_key_takes_precedence(monkeypatch):
    openai = Mock()
    monkeypatch.setattr("gap_finder.language_model.client.OpenAI", openai)
    monkeypatch.setenv("GAP_FINDER_API_KEY", "environment-key")

    OpenAIClient(model_name="test-model", api_key="explicit-key")

    openai.assert_called_once_with(base_url=None, api_key="explicit-key")


def test_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GAP_FINDER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")

    with pytest.raises(MissingAPIKeyError, match="GAP_FINDER_API_KEY"):
        OpenAIClient(model_name="test-model")


class TestOpenAIClientInvoke:
    def test_returns_message_content_from_openai_response(self):
        client = OpenAIClient.__new__(OpenAIClient)
        client.model_name = "test-model"
        client.client = Mock()
        client.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A precise answer."),
                )
            ]
        )

        response = client.invoke(
            prompt="What is a research gap?",
            system_prompt="Answer briefly.",
        )

        assert response == "A precise answer."
        client.client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is a research gap?"},
            ],
        )

    def test_returns_none_when_openai_response_content_is_none(self):
        client = OpenAIClient.__new__(OpenAIClient)
        client.model_name = "test-model"
        client.client = Mock()
        client.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None),
                )
            ]
        )

        response = client.invoke(
            prompt="What is missing?",
            system_prompt="Answer briefly.",
        )

        assert response is None
        client.client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is missing?"},
            ],
        )

    def test_wraps_client_exception_in_client_error(self):
        original_error = RuntimeError("request failed")
        client = OpenAIClient.__new__(OpenAIClient)
        client.model_name = "test-model"
        client.client = Mock()
        client.client.chat.completions.create.side_effect = original_error

        with pytest.raises(ClientError, match="request failed") as error:
            client.invoke(prompt="Prompt", system_prompt="System prompt")

        assert error.value.__cause__ is original_error
