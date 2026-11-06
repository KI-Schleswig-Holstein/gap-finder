import numpy as np

from gap_finder.embedding_model.embedding_model import EmbeddingModel


class DummyEmbeddingModel(EmbeddingModel):
    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_embedding_model_keeps_bertopic_compatible_embed_method():
    model = DummyEmbeddingModel()

    np.testing.assert_array_equal(
        model.embed(["first", "second"]),
        np.ones((2, 2), dtype=np.float32),
    )
