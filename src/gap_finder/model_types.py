"""Dependency-light public input types for model configuration."""

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.language_model.client import OpenAIClient
from gap_finder.language_model.language_model import LanguageModel


@runtime_checkable
class SentenceTransformerLike(Protocol):
    """Runtime shape used by a SentenceTransformer supplied by the caller."""

    model_name_or_path: str
    device: Any

    def encode(
        self,
        sentences: list[str],
        *,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> Any: ...

    def to(self, device: Any) -> Any: ...


@runtime_checkable
class TextGenerationPipelineLike(Protocol):
    """Runtime shape used by a Transformers text-generation pipeline."""

    task: str
    tokenizer: Any
    model: Any

    def __call__(self, inputs: Any, **kwargs: Any) -> Any: ...


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers.pipelines.text_generation import TextGenerationPipeline

    EmbeddingModelInput: TypeAlias = (
        str | SentenceTransformer | OpenAIEmbeddingClient | EmbeddingModel
    )
    LanguageModelInput: TypeAlias = (
        str | OpenAIClient | TextGenerationPipeline | LanguageModel
    )
    TopicEmbeddingModelInput: TypeAlias = str | SentenceTransformer
else:
    EmbeddingModelInput: TypeAlias = (
        str | SentenceTransformerLike | OpenAIEmbeddingClient | EmbeddingModel
    )
    LanguageModelInput: TypeAlias = (
        str | OpenAIClient | TextGenerationPipelineLike | LanguageModel
    )
    TopicEmbeddingModelInput: TypeAlias = str | SentenceTransformerLike


__all__ = [
    "EmbeddingModelInput",
    "LanguageModelInput",
    "SentenceTransformerLike",
    "TextGenerationPipelineLike",
    "TopicEmbeddingModelInput",
]
