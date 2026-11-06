"""Examples of the supported ways to initialize a GapFinder."""

from pathlib import Path

from gap_finder import GapFinder
from gap_finder.embedding_model import OpenAIEmbeddingClient, create_embedding_model
from gap_finder.vector_store import NumpyVectorStore

API_BASE_URL = "http://localhost:11434/v1"
LANGUAGE_MODEL = "qwen3:latest"


def from_pdf() -> GapFinder:
    return GapFinder.from_pdf(
        folder_path="examples/data/pdf",
        language_model=LANGUAGE_MODEL,
        base_url=API_BASE_URL,
        api_key="-",
    )


def from_chunks() -> GapFinder:
    return GapFinder.from_chunks(
        chunks=[
            "A knowledge gap is a topic that is not adequately covered.",
            "Questions can be compared with a corpus using embeddings.",
        ],
        language_model=LANGUAGE_MODEL,
        base_url=API_BASE_URL,
        api_key="-",
    )


def from_vector_store() -> GapFinder:
    embedding_model = create_embedding_model(
        OpenAIEmbeddingClient(
            model_name="qwen3-embedding:8b",
            base_url=API_BASE_URL,
            api_key="-",
        )
    )
    chunks = ["An already embedded corpus can be supplied directly."]
    embeddings = embedding_model.encode(chunks)
    vector_store = NumpyVectorStore(dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    return GapFinder(
        vector_store=vector_store,
        embedding_model=embedding_model,
        language_model=LANGUAGE_MODEL,
        base_url=API_BASE_URL,
        api_key="-",
    )


def from_snapshot(snapshot_path: str | Path) -> GapFinder:
    return GapFinder.load_model(snapshot_path, api_key="-")


if __name__ == "__main__":
    gap_finder = from_chunks()
    print(gap_finder.find().head())
