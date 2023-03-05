import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class NLTKSentimentAnalyzer:
    """
    A class for performing sentiment analysis on a given text using NLTK library.

    Example:
     analyzer = NLTKSentimentAnalyzer(text="I love Python!")
     result = analyzer.analyze_sentiment()
     print(result)
    {'polarity': 0.6369, 'subjectivity': None}
    """

    def __init__(self, text: str = None, file_path: str = None) -> None:
        """
        Initializes the NLTKSentimentAnalyzer object with the given text or file path.

        Args:
            text: The text to be analyzed.
            file_path: The path to the text file to be analyzed.
        """
        self.text = None
        self.analyzer = SentimentIntensityAnalyzer()

        if text is not None:
            self.set_text(text)
        elif file_path is not None:
            self.set_file_path(file_path)

    def set_text(self, text: str) -> None:
        """
        Sets the text to be analyzed.

        Args:
            text: The text to be analyzed.
        """
        self.text = text

    def set_file_path(self, file_path: str) -> None:
        """
        Sets the filepath to a text file to be analyzed.

        Args:
            file_path: The path to the text file to be analyzed.
        """
        with open(file_path, 'r') as file:
            self.text = file.read()

    def analyze_sentiment(self) -> dict:
        """
        Analyzes the sentiment of the text using NLTK library.

        Returns:
            A dictionary containing the polarity and subjectivity scores of the text.
        """
        sentiment_scores = self.analyzer.polarity_scores(self.text)
        sentiment_polarity = sentiment_scores['compound']
        sentiment_subjectivity = None  # NLTK does not provide a subjectivity score
        return {'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity}


class VaderAnalyzer:
    """
    A class for performing sentiment analysis using the Vader library.
    """

    def __init__(self, text: str = None, file_path: str = None) -> None:
        """
        Initializes the VaderAnalyzer object with the given text or file path.

        Args:
            text: The text to be analyzed.
            file_path: The path to the text file to be analyzed.
        """
        self.text = None
        self.file_path = None
        self.analyzer = SentimentIntensityAnalyzer()

        if text is not None:
            self.set_text(text)
        elif file_path is not None:
            self.set_file_path(file_path)

    def set_text(self, text: str) -> None:
        """
        Sets the text to be analyzed.

        Args:
            text: The text to be analyzed.
        """
        self.text = text

    def set_file_path(self, file_path: str) -> None:
        """
        Sets the filepath to a text file to be analyzed.

        Args:
            file_path: The path to the text file to be analyzed.
        """
        with open(file_path, 'r') as file:
            self.text = file.read()
        self.file_path = file_path

    def analyze(self) -> dict:
        """
        Performs sentiment analysis on the text using Vader.

        Returns:
            A dictionary of sentiment scores for the text.
        """
        if self.text is None:
            raise ValueError("No text or filepath set.")

        scores = self.analyzer.polarity_scores(self.text)
        return scores
