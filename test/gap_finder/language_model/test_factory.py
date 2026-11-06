import subprocess
import sys
from unittest.mock import Mock

import pytest
from transformers.pipelines.text_generation import TextGenerationPipeline

from gap_finder.language_model import factory
from gap_finder.language_model.client import OpenAIClient
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.language_model.local_language_model import LocalLanguageModel
from gap_finder.language_model.openai_language_model import OpenAILanguageModel


def test_string_creates_configured_openai_model(monkeypatch):
    client = OpenAIClient.__new__(OpenAIClient)
    constructor = Mock(return_value=client)
    monkeypatch.setattr("gap_finder.language_model.client.OpenAIClient", constructor)

    result = factory.create_language_model(
        "remote-model",
        base_url="http://example.test/v1",
        api_key="secret",
    )

    assert isinstance(result, OpenAILanguageModel)
    assert result.client is client
    constructor.assert_called_once_with(
        model_name="remote-model",
        base_url="http://example.test/v1",
        api_key="secret",
    )


def test_reuses_configured_openai_client():
    client = OpenAIClient.__new__(OpenAIClient)

    result = factory.create_language_model(
        client,
        base_url="ignored",
        api_key="ignored",
    )

    assert isinstance(result, OpenAILanguageModel)
    assert result.client is client


def test_reuses_text_generation_pipeline():
    pipeline = TextGenerationPipeline.__new__(TextGenerationPipeline)
    pipeline.task = "text-generation"
    pipeline.tokenizer = Mock()

    result = factory.create_language_model(pipeline)

    assert isinstance(result, LocalLanguageModel)
    assert result.client is pipeline


def test_rejects_unsupported_model():
    with pytest.raises(TypeError, match="language_model"):
        factory.create_language_model(object())


def test_returns_existing_language_model_unchanged():
    language_model = Mock(spec=LanguageModel)

    result = factory.create_language_model(language_model)

    assert result is language_model


def test_api_model_does_not_import_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from gap_finder.language_model.client import OpenAIClient
from gap_finder.language_model.factory import create_language_model

create_language_model(OpenAIClient.__new__(OpenAIClient))
assert "transformers" not in sys.modules
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
