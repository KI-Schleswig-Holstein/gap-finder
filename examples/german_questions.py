"""Generate German, domain-specific questions with an OpenAI-compatible API."""

from gap_finder import GapFinder


def main():
    gap_finder = GapFinder.from_chunks(
        chunks=[
            "Neuronale Netze lernen Muster aus Trainingsdaten.",
            "Die Qualität der Daten beeinflusst die Qualität des Modells.",
            "Modelle sollten auf unabhängigen Testdaten evaluiert werden.",
        ],
        language_model="qwen3:latest",
        base_url="http://localhost:11434/v1",  # Ollama API base URL
        api_key="-",
        language="german",
        domain_specific_questions=True,
        domain_description=(
            "Maschinelles Lernen mit Schwerpunkt auf Datenqualität und "
            "Modellevaluation."
        ),
    )
    topics = gap_finder.find()
    print(topics.head())


if __name__ == "__main__":
    main()
