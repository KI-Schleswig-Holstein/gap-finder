from __future__ import annotations

from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.model_types import EmbeddingModelInput


def create_embedding_model(
    embedding_model: EmbeddingModelInput,
) -> EmbeddingModel:
    """Create an embedding wrapper from a model name or configured client."""
    if isinstance(embedding_model, EmbeddingModel):
        return embedding_model

    if isinstance(embedding_model, OpenAIEmbeddingClient):
        from gap_finder.embedding_model.openai_embedding_model import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel(client=embedding_model)
    if isinstance(embedding_model, str):
        from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel

        return LocalEmbeddingModel(client=embedding_model)

    from sentence_transformers import SentenceTransformer

    if isinstance(embedding_model, SentenceTransformer):
        from gap_finder.embedding_model.local_embedding_model import LocalEmbeddingModel

        return LocalEmbeddingModel(client=embedding_model)
    raise TypeError(
        "embedding_model must be an EmbeddingModel, model name, "
        "SentenceTransformer, or "
        f"OpenAIEmbeddingClient; got {type(embedding_model).__name__}."
    )
