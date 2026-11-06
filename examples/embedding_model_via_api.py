from gap_finder.embedding_model import OpenAIEmbeddingClient
from gap_finder.main import GapFinder


def main():
    embedding_model = OpenAIEmbeddingClient(
        model_name="qwen3-embedding:8b",
        base_url="http://localhost:11434/v1",
        api_key="-"
    )

    gap_finder = GapFinder.from_pdf(
        folder_path="examples/data/pdf",
        embedding_model=embedding_model,
        language_model="qwen3:latest",
        language="english",
        domain_description="The domain is machine learning.",
        base_url="http://localhost:11434/v1",
        api_key="-",
    )
    topics = gap_finder.find()
    print(topics.head())



if __name__ == "__main__":
    main()
