from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Common interface for local and API-backed text embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text into an embedding matrix."""
        raise NotImplementedError

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Embed documents using BERTopic-compatible arguments."""
        return self.encode(
            documents,
            show_progress_bar=verbose,
            batch_size=32,
        )

    def activate(self) -> None:
        """Make internally owned resources ready for use."""
        return

    def deactivate(self) -> None:
        """Optionally offload internally owned resources between steps."""
        return

    def cleanup(self) -> None:
        """Release internally owned resources after the pipeline finishes."""
        return
