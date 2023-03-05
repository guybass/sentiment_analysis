import spacy
from typing import Dict, List, Union
from collections import defaultdict
from textblob import TextBlob



class SpacyAnalyzer:
    """
    A class for performing sentiment analysis using the Spacy library.
    """
    def init(self, model_name: str = "en_core_web_sm", text: str = None, filepath: str = None) -> None:
        """
            Initializes the analyzer with either text or a filepath to a text file.

            Args:
            - model_name: A string representing the name of the Spacy model to be used.
            - text: A string representing the text to be analyzed.
            - filepath: A string representing the path of the file to be analyzed.
        """
        self.text: Union[str, None] = None
        self.filepath: Union[str, None] = None
        self.nlp = spacy.load(model_name)

        if text is not None:
            self.set_text(text)
        elif filepath is not None:
            self.set_filepath(filepath)

    def set_text(self, text: str) -> None:
        """
           Sets the text to be analyzed.

            Args:
                - text: A string representing the text to be analyzed.
        """
        self.text = text

    def set_filepath(self, filepath: str) -> None:
        """
        Sets the filepath to a text file to be analyzed.

         Args:
        - filepath: A string representing the path of the file to be analyzed.
        """
        with open(filepath, 'r') as f:
            self.text = f.read()
        self.filepath = filepath

    def analyze(self) -> Dict[str, float]:
        """
        Performs sentiment analysis on the text using Spacy.

        Returns:
        - A dictionary containing the sentiment score for the text.
        """
        if self.text is None:
            raise ValueError("No text or filepath set.")

        doc = self.nlp(self.text)
        pos_score = doc.sentiment.polarity
        return {"pos_score": pos_score}


class EntitySentimentAnalyzer:
    def init(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, text: str) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Analyzes the sentiment and entities in the given text.

        Args:
        - text: A string representing the text to analyze.

        Returns:
        - A dictionary with two keys:
          - "sentiment": The sentiment score of the text (between -1 and 1).
          - "entities": A dictionary where each key is an entity mentioned in the text,
            and each value is a sentiment score for that entity (between -1 and 1).
        """
        doc = self.nlp(text)
        entity_sentiments = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                entity_sentiments[ent.text].append(ent.sentiment.polarity)

        entities: Dict[str, float] = {}
        for entity, sentiments in entity_sentiments.items():
            entities[entity] = sum(sentiments) / len(sentiments)

        return {
            "sentiment": TextBlob(text).sentiment.polarity,
            "entities": entities
        }

