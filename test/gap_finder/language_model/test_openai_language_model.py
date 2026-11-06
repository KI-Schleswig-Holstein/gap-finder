from unittest.mock import Mock, call

import pytest

from gap_finder.language_model.client import ClientError
from gap_finder.language_model.openai_language_model import OpenAILanguageModel


class TestOpenAILanguageModelInvoke:
    def test_delegates_to_client_and_returns_response(self):
        client = Mock()
        client.invoke.return_value = "A remote answer."
        language_model = OpenAILanguageModel(client=client)

        response = language_model.invoke(
            prompt="Explain the gap.",
            system_prompt="Be concise.",
        )

        assert response == "A remote answer."
        client.invoke.assert_called_once_with(
            prompt="Explain the gap.",
            system_prompt="Be concise.",
        )

    def test_raises_client_error_when_client_returns_none(self):
        client = Mock()
        client.invoke.return_value = None
        language_model = OpenAILanguageModel(client=client)

        with pytest.raises(ClientError, match="returned None response") as error:
            language_model.invoke(
                prompt="Explain the gap.",
                system_prompt="Be concise.",
            )

        assert error.value.context == {
            "prompt": "Explain the gap.",
            "system_prompt": "Be concise.",
            "client": client,
        }


class TestOpenAILanguageModelInvokeBatch:
    def test_delegates_each_prompt_to_client_and_returns_responses(self):
        client = Mock()
        client.invoke.side_effect = ["First answer.", "Second answer."]
        language_model = OpenAILanguageModel(client=client)

        responses = language_model.invoke_batch(
            prompts=["Explain the first gap.", "Explain the second gap."],
            system_prompt="Be concise.",
        )

        assert responses == ["First answer.", "Second answer."]
        client.invoke.assert_has_calls(
            [
                call(prompt="Explain the first gap.", system_prompt="Be concise."),
                call(prompt="Explain the second gap.", system_prompt="Be concise."),
            ]
        )

    def test_raises_client_error_when_batch_response_is_none(self):
        client = Mock()
        client.invoke.side_effect = ["First answer.", None]
        language_model = OpenAILanguageModel(client=client)

        with pytest.raises(ClientError, match="returned None response") as error:
            language_model.invoke_batch(
                prompts=["Explain the first gap.", "Explain the second gap."],
                system_prompt="Be concise.",
            )

        assert error.value.context == {
            "prompt": "Explain the second gap.",
            "system_prompt": "Be concise.",
            "client": client,
            "index": 1,
        }
