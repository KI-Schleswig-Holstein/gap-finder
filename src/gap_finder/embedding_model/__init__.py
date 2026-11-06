from importlib import import_module
from typing import TYPE_CHECKING

from gap_finder.embedding_model.client import OpenAIEmbeddingClient

if TYPE_CHECKING:
    from gap_finder.embedding_model.embedding_model import EmbeddingModel
    from gap_finder.embedding_model.factory import create_embedding_model
    from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel
    from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel

__all__ = [
    "EmbeddingModel",
    "LocalEmbeddingModel",
    "OpenAIEmbeddingClient",
    "OpenAIEmbeddingModel",
    "create_embedding_model",
]


def __getattr__(name: str):
    if name == "EmbeddingModel":
        from gap_finder.embedding_model.embedding_model import EmbeddingModel

        return EmbeddingModel
    if name == "create_embedding_model":
        from gap_finder.embedding_model.factory import create_embedding_model

        return create_embedding_model
    if name == "LocalEmbeddingModel":
        from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel

        return LocalEmbeddingModel
    if name == "OpenAIEmbeddingModel":
        from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel
    if name == "factory":
        return import_module("gap_finder.embedding_model.factory")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
