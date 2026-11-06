import numpy as np
from tqdm import tqdm

from gap_finder.embedding_model.client import OpenAIEmbeddingClient
from gap_finder.embedding_model.embedding_model import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model backed by an OpenAI-compatible embeddings API."""

    def __init__(
        self,
        client: OpenAIEmbeddingClient,
    ) -> None:
        super().__init__()
        self.model_name = client.model_name
        self.client = client.client

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings: list[list[float]] = []
        batch_starts = range(0, len(texts), batch_size)
        for start in tqdm(batch_starts, disable=not show_progress_bar):
            batch = texts[start:start + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            response_data = list(response.data)
            if all(hasattr(item, "index") for item in response_data):
                response_data.sort(key=lambda item: item.index)
            embeddings.extend(item.embedding for item in response_data)

        return np.asarray(embeddings, dtype=np.float32)
