from __future__ import annotations

from gap_finder.language_model import client as language_model_client
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.model_types import LanguageModelInput


def create_language_model(
    language_model: LanguageModelInput,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LanguageModel:
    """Create a language-model wrapper from a model name or client."""
    if isinstance(language_model, LanguageModel):
        return language_model

    if isinstance(language_model, str):
        from gap_finder.language_model.openai_language_model import OpenAILanguageModel

        client = language_model_client.OpenAIClient(
            model_name=language_model,
            base_url=base_url,
            api_key=api_key,
        )
        return OpenAILanguageModel(client=client)
    if isinstance(language_model, language_model_client.OpenAIClient):
        from gap_finder.language_model.openai_language_model import OpenAILanguageModel

        return OpenAILanguageModel(client=language_model)

    from transformers.pipelines.text_generation import TextGenerationPipeline

    if isinstance(language_model, TextGenerationPipeline):
        from gap_finder.language_model.local_language_model import LocalLanguageModel

        return LocalLanguageModel(client=language_model)
    raise TypeError(
        "language_model must be a LanguageModel, model name, OpenAIClient, or "
        f"TextGenerationPipeline; got {type(language_model).__name__}."
    )
