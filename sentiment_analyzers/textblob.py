import requests
from typing import Dict
from textblob import TextBlob


class TextBlobSentimentAnalyzer:
    """
    A class for performing sentiment analysis on a given text using the TextBlob library.

    Example:
     analyzer = TextBlobSentimentAnalyzer(text="I love Python!")
     result = analyzer.analyze_sentiment()
     print(result)
    {'polarity': 0.5, 'subjectivity': 0.6}
    """

    def __init__(self, text: str = None, file_path: str = None) -> None:
        """
        Initializes the TextBlobSentimentAnalyzer object with the given text or file path.

        Args:
            text (str): The text to be analyzed. Default is None.
            file_path (str): The path to the text file to be analyzed. Default is None.
        """
        if text is not None:
            self.text = text
        elif file_path is not None:
            with open(file_path, 'r') as f:
                self.text = f.read()

    def analyze_sentiment(self) -> Dict[str, float]:
        """
        Analyzes the sentiment of the text using TextBlob library.

        Returns:
            dict: A dictionary containing the polarity and subjectivity scores of the text.
        """
        blob = TextBlob(self.text)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        return {'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity}


# Extracting sentiment scores from news articles using TextBlob:
class NewsSentimentAnalyzer:
    """
    A class for performing sentiment analysis on news articles using TextBlob.

    Example:
        analyzer = NewsSentimentAnalyzer(url="https://www.example.com")
        score = analyzer.get_sentiment_score()
        print(score)

    Attributes:
        url: str
            The URL of the news article to analyze.
        text: str
            The text of the news article extracted from the URL.
    """
    def __init__(self, url: str):
        """
        Initializes the NewsSentimentAnalyzer object with the given URL.

        Args:
            url: The URL of the news article to analyze.
        """
        self.url = url
        self.text = self.get_text_from_url()

    def get_text_from_url(self) -> str:
        """
        Gets the text from the given URL.

        Returns:
            The text of the news article extracted from the URL.
        """
        response = requests.get(self.url)
        return response.text

    def get_sentiment_score(self) -> float:
        """
        Gets the sentiment score of the news article.

        Returns:
            The sentiment score of the news article as a float.
        """
        blob = TextBlob(self.text)
        return blob.sentiment.polarity

    def get_sentiment_label(self) -> str:
        """
        Gets the sentiment label of the news article.

        Returns:
            The sentiment label of the news article as a string, which is either 'positive', 'negative', or 'neutral'.
        """
        score = self.get_sentiment_score()
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'


# Tracking changes in consumer sentiment over time using TextBlob and matplotlib:
class ConsumerSentimentAnalysis:
    """
    A class for tracking changes in consumer sentiment over time using TextBlob and matplotlib.
    """

    def __init__(self, urls: list[str]):
        """
        Initialize the ConsumerSentimentAnalysis object with a list of URLs.

        Args:
            urls (List[str]): List of URLs to retrieve text content from
        """
        self.urls = urls
        self.sentiment_scores = self.get_sentiment_scores()

    def get_text_from_url(self, url: str) -> str:
        """
        Retrieve the text content from a given URL.

        Args:
            url (str): URL to retrieve text content from

        Returns:
            str: Text content of the URL
        """
        response = requests.get(url)
        return response.text

    def get_sentiment_score(self, text: str) -> float:
        """
        Calculate the sentiment score of a given text using TextBlob.

        Args:
            text (str): Text to analyze

        Returns:
            float: Sentiment score
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def get_sentiment_scores(self) -> list[float]:
        """
        Retrieve the sentiment scores for each URL.

        Returns:
            List[float]: List of sentiment scores
        """
        sentiment_scores = []
        for url in self.urls:
            text = self.get_text_from_url(url)
            sentiment_score = self.get_sentiment_score(text)
            sentiment_scores.append(sentiment_score)
        return sentiment_scores


