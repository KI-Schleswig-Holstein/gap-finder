from sentence_transformers import SentenceTransformer
from transformers import pipeline

from gap_finder.main import GapFinder


def main():
    language_model = pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-4B",
        device="cuda",
        max_new_tokens=1024,
    )
    embedding_model = SentenceTransformer(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )

    gap_finder = GapFinder.from_pdf(
        folder_path="examples/data/pdf",
        embedding_model=embedding_model,
        topic_embedding_model=embedding_model,
        language_model=language_model,
        question_generation_batch_size=16,
        language="english",
        summarize_chunks=True,
        domain_description="The domain is machine learning.",
        cleanup_models_after_usage=True,
    )
    topics = gap_finder.find()
    print(topics.head())



if __name__ == "__main__":
    main()
