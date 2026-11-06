from pathlib import Path


def partition_pdf(**kwargs):
    """Import the optional PDF runtime only when a PDF is actually parsed."""
    from unstructured.partition.pdf import partition_pdf as unstructured_partition_pdf

    return unstructured_partition_pdf(**kwargs)


def _create_text_splitter(chunk_size: int, chunk_overlap: int):
    """Import the text-splitting runtime only when chunking is requested."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


class PDFChunker:
    """Splits a PDF's text content into overlapping character chunks."""

    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 600, languages: list[str] | None = None) -> None:
        self.languages = ["eng"] if languages is None else languages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = None

    def chunk_pdf(self, file_path: Path) -> list[str]:
        """Load `file_path` and split its text content into chunks."""
        elements = partition_pdf(filename=str(file_path), strategy="fast", languages=self.languages)
        text = "\n\n".join(element.text for element in elements if element.text)
        if self.splitter is None:
            self.splitter = _create_text_splitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self.splitter.split_text(text)
