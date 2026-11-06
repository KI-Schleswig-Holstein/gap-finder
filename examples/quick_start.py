from gap_finder.main import GapFinder


def main():
    gap_finder = GapFinder.from_pdf(
        folder_path="examples/data/pdf",
        base_url="http://localhost:11434/v1",  # Ollama API base URL
        api_key="-",
        domain_description="The domain is machine learning.",
    )
    topics = gap_finder.find()
    print(topics.head())



if __name__ == "__main__":
    main()
