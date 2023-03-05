from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


from typing import Dict, List, Union
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VADERSentimentAnalyzer:
    """
    A class to analyze sentiment using the VADER (Valence Aware Dictionary and Sentiment Reasoner) algorithm.
    """

    def __init__(self, text: str = None, file_path: str = None):
        """
        Initialize the VADERSentimentAnalyzer.

        Args:
            text (str): The text to analyze.
            file_path (str): The path to a file containing the text to analyze.
        """
        self.text = None
        self.analyzer = SentimentIntensityAnalyzer()

        if text is not None:
            self.set_text(text)
        elif file_path is not None:
            self.set_file_path(file_path)

    def set_text(self, text: str) -> None:
        """
        Set the text to be analyzed.

        Args:
            text (str): The text to analyze.
        """
        self.text = text

    def set_file_path(self, file_path: str) -> None:
        """
        Set the path to the file containing the text to analyze.

        Args:
            file_path (str): The path to a file containing the text to analyze.
        """
        with open(file_path, 'r') as f:
            self.text = f.read()

    def analyze_sentiment(self) -> Dict[str, Union[float, None]]:
        """
        Analyze the sentiment of the text.

        Returns:
            dict: A dictionary containing the polarity and subjectivity scores.
        """
        sentiment_scores = self.analyzer.polarity_scores(self.text)
        sentiment_polarity = sentiment_scores['compound']
        sentiment_subjectivity = None  # VADER does not provide a subjectivity score
        return {'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity}

    def analyze_sentiment_by_sentence(self) -> List[Dict[str, Union[float, None]]]:
        """
        Analyze the sentiment of each sentence in the text.

        Returns:
            list: A list of dictionaries containing the polarity and subjectivity scores for each sentence.
        """
        sentences = self.text.split('.')
        sentiment_scores_list = []
        for sentence in sentences:
            sentiment_scores = self.analyzer.polarity_scores(sentence)
            sentiment_polarity = sentiment_scores['compound']
            sentiment_subjectivity = None  # VADER does not provide a subjectivity score
            sentiment_scores_list.append({'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity})
        return sentiment_scores_list

    def analyze_sentiment_by_category(self) -> Dict[str, Union[Dict[str, int], Dict[str, float]]]:
        """
        Analyze the sentiment of the text by category (positive, negative, neutral).

        Returns:
            dict: A dictionary containing the count and percentage of each sentiment category.
        """
        sentiment_categories = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = self.analyzer.polarity_scores(self.text)
        sentiment_polarity = sentiment_scores['compound']
        if sentiment_polarity >= 0.05:
            sentiment_categories['positive'] += 1
        elif sentiment_polarity <= -0.05:
            sentiment_categories['negative'] += 1
        else:
            sentiment_categories['neutral'] += 1
        total_count = sum(sentiment_categories.values())
        sentiment_percentages = {k: round(v/total_count*100, 2) for k, v in sentiment_categories.items()}
        return {'count': sentiment_categories, 'percentages': sentiment_percentages}

