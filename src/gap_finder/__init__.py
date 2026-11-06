from typing import TYPE_CHECKING

from gap_finder._auth import MissingAPIKeyError
from gap_finder.model_types import (
    EmbeddingModelInput,
    LanguageModelInput,
    SentenceTransformerLike,
    TextGenerationPipelineLike,
    TopicEmbeddingModelInput,
)

if TYPE_CHECKING:
    from gap_finder.main import GapFinder

__all__ = [
    "EmbeddingModelInput",
    "GapFinder",
    "LanguageModelInput",
    "MissingAPIKeyError",
    "SentenceTransformerLike",
    "TextGenerationPipelineLike",
    "TopicEmbeddingModelInput",
]


def __getattr__(name: str):
    if name == "GapFinder":
        from gap_finder.main import GapFinder

        return GapFinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
