"""Regression tests for keeping unit-test collection lightweight."""

import subprocess
import sys
from pathlib import Path


def test_unit_test_configuration_does_not_load_integration_runtimes():
    test_dir = Path(__file__).parent
    script = f"""
import sys

sys.path.insert(0, {str(test_dir)!r})

import conftest
import gap_finder
import gap_finder.main
import gap_finder.topic_generation
import gap_finder.vector_store.pdf_chunker

forbidden_modules = (
    "integration",
    "langchain_text_splitters",
    "unstructured",
    "cv2",
    "bertopic",
    "sentence_transformers",
    "torch",
    "transformers",
)
loaded_modules = [
    module_name
    for module_name in forbidden_modules
    if module_name in sys.modules
]
assert loaded_modules == [], loaded_modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)
