from typing import Dict, List
from textblob import TextBlob


def extract_sentiment_scores(text: str) -> Dict[str, float]:
    """
    Extracts the sentiment scores for a given text using TextBlob.

    Args:
        text (str): The text to extract the sentiment scores from.

    Returns:
        A dictionary containing the sentiment scores for the text. The dictionary includes the following keys:
        - polarity: A float value between -1 and 1 indicating the sentiment polarity, where negative values indicate negative sentiment, positive values indicate positive sentiment, and 0 indicates neutral sentiment.
        - subjectivity: A float value between 0 and 1 indicating the subjectivity of the text, where 0 is very objective and 1 is very subjective.
    """
    blob = TextBlob(text)
    sentiment = {'polarity': blob.sentiment.polarity,
                 'subjectivity': blob.sentiment.subjectivity}
    return sentiment


def extract_pos_tags(text: str) -> List[str]:
    """
    Extracts the POS tags for a given text using TextBlob.

    Args:
        text (str): The text to extract the POS tags from.

    Returns:
        A list of POS tags for the text.
    """
    blob = TextBlob(text)
    pos_tags = [tag[1] for tag in blob.tags]
    return pos_tags


def extract_named_entities(text: str) -> List[str]:
    """
    Extracts the named entities for a given text using TextBlob.

    Args:
        text (str): The text to extract the named entities from.

    Returns:
        A list of named entities for the text.
    """
    blob = TextBlob(text)
    named_entities = [str(entity) for entity in blob.noun_phrases]
    return named_entities


def extract_ngrams(text: str, n: int = 3) -> List[str]:
    """
    Extracts the n-grams for a given text using TextBlob.

    Args:
        text (str): The text to extract the n-grams from.
        n (int): The value of n in n-grams. Default is 3.

    Returns:
        A list of n-grams for the text.
    """
    blob = TextBlob(text)
    ngrams = blob.ngrams(n)
    ngrams = [' '.join(grams) for grams in ngrams]
    return ngrams


def prepare_text_for_embedding(text: str) -> Dict[str, List[str]]:
    """
    Prepares a given text for embedding by extracting various attributes and scores using TextBlob.

    Args:
        text (str): The text to prepare.

    Returns:
        A dictionary containing the extracted attributes and scores. The dictionary includes the following keys:
        - sentiment: A dictionary containing the sentiment scores for the text. See extract_sentiment_scores()
        - pos_tags: A list of POS tags for the text. See extract_pos_tags()
        - named_entities: A list of named entities for the text. See extract_named_entities()
        - ngrams: A list of n-grams for the text. See extract_ngrams()
    """
    sentiment = extract_sentiment_scores(text)
    pos_tags = extract_pos_tags(text)
    named_entities = extract_named_entities(text)
    ngrams = extract_ngrams(text)
    return {'sentiment': sentiment, 'pos_tags': pos_tags, 'named_entities': named_entities, 'ngrams': ngrams}


def get_subjectivity(text: str) -> float:
    """
    Computes the subjectivity score of a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The subjectivity score, which ranges from 0 (objective) to 1 (subjective).
    """
    blob = TextBlob(text)
    return blob.sentiment.subjectivity


def get_polarity(text: str) -> float:
    """
    Computes the polarity score of a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The polarity score, which ranges from -1 (negative) to 1 (positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_sentiment(text: str) -> Tuple[float, float]:
    """
    Computes both the subjectivity and polarity scores of a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        Tuple[float, float]: A tuple of the form (polarity, subjectivity).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_noun_phrases(text: str) -> List[str]:
    """
    Extracts all noun phrases from a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        List[str]: A list of all noun phrases found in the text.
    """
    blob = TextBlob(text)
    return blob.noun_phrases


def get_word_counts(text: str) -> dict:
    """
    Computes the frequency of each word in a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary where the keys are words in the text and the values are their frequency.
    """
    blob = TextBlob(text)
    return blob.word_counts


def get_word_frequencies(text: str) -> dict:
    """
    Computes the frequency of each word in a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary where the keys are words in the text and the values are their frequency.
    """
    blob = TextBlob(text)
    return blob.word_counts


def get_noun_counts(text: str) -> dict:
    """
    Computes the frequency of each noun in a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary where the keys are nouns in the text and the values are their frequency.
    """
    blob = TextBlob(text)
    return blob.noun_counts


def get_noun_frequencies(text: str) -> dict:
    """
    Computes the frequency of each noun in a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary where the keys are nouns in the text and the values are their frequency.
    """
    blob = TextBlob(text)
    return blob.noun_counts


def get_noun_phrases_counts(text: str) -> dict:
    """
    Computes the frequency of each noun phrase in a given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary where the keys are noun phrases in the text and the values are their frequency.
    """
    blob = TextBlob(text)
    return blob.np_counts
