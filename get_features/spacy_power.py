import spacy
from typing import Dict, Any

nlp = spacy.load("en_core_web_sm")


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyzes the given text using the spaCy en_core_web_sm model and returns various
    attributes, scores, and other information.

    Args:
        text (str): The text to analyze.

    Returns:
        A dictionary containing the following keys and values:
        - "text": the original text
        - "words": a list of individual word strings
        - "lemmas": a list of lemmas for each word
        - "pos_tags": a list of part-of-speech tags for each word
        - "entities": a list of named entities in the text and their labels
        - "sentences": a list of sentence strings
        - "polarity": a float representing the overall polarity score (-1 to 1)
        - "subjectivity": a float representing the overall subjectivity score (0 to 1)
    """
    doc = nlp(text)

    # Get individual word tokens and lemmas
    words = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]

    # Get part-of-speech tags for each word
    pos_tags = [token.pos_ for token in doc]

    # Get named entities and their labels
    entities = [(entity.text, entity.label_) for entity in doc.ents]

    # Get individual sentence strings
    sentences = [sent.text.strip() for sent in doc.sents]

    # Calculate polarity and subjectivity scores
    polarity = doc.sentiment.polarity
    subjectivity = doc.sentiment.subjectivity

    # Return a dictionary containing all the analyzed attributes
    return {
        "text": text,
        "words": words,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities,
        "sentences": sentences,
        "polarity": polarity,
        "subjectivity": subjectivity
    }
