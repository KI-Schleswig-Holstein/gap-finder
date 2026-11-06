"""
This module defines several *unchanging* values used *repeatably* throughout the
project. It may be expanded or reduced as see fits.

Tips for expanding this module with new constants:

- Create a new class for each category of constants with class attributes for
  each constant
- Use only uppercase letters for constant names
- Make your constant classes (and/or single constants) the only *public* objects
  of this module. Delete all other objects.

Examples
--------
>>> import project_name.constants as constants
>>> # get Path object for the project root directory
>>> constants.Paths.ROOT
"""
from pathlib import Path


_ROOT = Path(__file__).parents[3]

class DataFileNames:
    QUESTIONS = "questions.json"
    QUESTIONS_RANKINGS = "questions_rankings.json"
    QUESTIONS_WITH_TOPICS = "questions_topics.json"
    TOPICS_RANKINGS_JSON = "topics_rankings.json"
    TOPIC_EXPLANATIONS_JSON = "topic_explanations.json"
    TOPIC_RELEVANCE_SCORES_JSON = "topic_relevance_scores.json"
    REPORT_MD = "report.md"
    BERTOPIC_MODEL = "bertopic_question_model"


class PromptTemplates:
    SUMMARIZE = "summarize.txt.j2"
    QUESTION_GENERATION = "question_generation/main.txt.j2"
    TOPIC_EXPLANATION = "topic_explanation.txt.j2"
    HYDE = "hyde.txt.j2"
    TOPIC_RELEVANCE = "topic_relevance.txt.j2"


# Clean up
# Deleting all objects that are not part of the public API

del _ROOT
del Path
