import gc
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from gap_finder.embedding_model.embedding_model import EmbeddingModel


class LocalEmbeddingModel(EmbeddingModel):
    """Embedding model backed by a local SentenceTransformer."""

    def __init__(
        self,
        client: str | SentenceTransformer,
    ) -> None:
        super().__init__()
        self._owns_client = isinstance(client, str)
        self.model_name = (
            client
            if isinstance(client, str)
            else getattr(client, "model_name_or_path", None)
        )
        self.client: SentenceTransformer | None = (
            None if isinstance(client, str) else client
        )
        self._target_device: Any = None
        self._active = not self._owns_client
        if self.client is not None:
            self._target_device = getattr(self.client, "device", None)

    def activate(self) -> None:
        """Load or reactivate an internally owned SentenceTransformer."""
        if not self._owns_client or self._active:
            return
        if self.client is None:
            if not isinstance(self.model_name, str):
                raise ValueError("Local embedding model has no model name.")
            self.client = SentenceTransformer(self.model_name)
            self._target_device = getattr(self.client, "device", None)
        elif self._target_device is not None and hasattr(self.client, "to"):
            self.client.to(self._target_device)
        self._active = True

    def _require_client(self) -> SentenceTransformer:
        self.activate()
        if self.client is None:
            raise RuntimeError("Local embedding model could not be activated.")
        return self.client

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        embeddings = self._require_client().encode(
            texts,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def deactivate(self) -> None:
        """Offload an internally owned model while retaining it for reuse."""
        if not self._owns_client or self.client is None or not self._active:
            return

        try:
            import torch
        except ImportError:
            gc.collect()
            return

        if hasattr(self.client, "to"):
            self.client.to(torch.device("cpu"))
        self._active = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """Release an internally owned model after the pipeline finishes."""
        if not self._owns_client or self.client is None:
            return
        self.deactivate()
        self.client = None
        self._active = False
        gc.collect()
