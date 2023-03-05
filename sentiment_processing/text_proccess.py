import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text, use_lemmatizer=True, use_stemmer=False):
    """
    Preprocesses a given text by tokenizing, removing stop words, and lemmatizing/stemming.

    Args:
        text (str): The text to preprocess.
        use_lemmatizer (bool): Whether to use WordNet lemmatization or not.
        use_stemmer (bool): Whether to use Porter stemming or not.

    Returns:
        str: The preprocessed text.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize or stem the tokens
    if use_lemmatizer:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif use_stemmer:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    # Return the preprocessed text as a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the NLTK Vader sentiment analyzer.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The sentiment score, which ranges from -1 (negative) to 1 (positive).
    """
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Analyze the sentiment of the text
    scores = sid.polarity_scores(text)

    # Return the sentiment score
    return scores['compound']


def batch_analyze_sentiment(texts):
    """
    Analyzes the sentiment of a list of texts using the NLTK Vader sentiment analyzer.

    Args:
        texts (list): A list of strings to analyze.

    Returns:
        list: A list of sentiment scores, which range from -1 (negative) to 1 (positive).
    """
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Analyze the sentiment of each text in the list
    scores = [sid.polarity_scores(text)['compound'] for text in texts]

    # Return the sentiment scores as a list
    return scores


def predict_sentiment(texts, threshold=0.0):
    """
    Predicts the sentiment of a list of texts by applying a threshold to the sentiment scores.

    Args:
        texts (list): A list of strings to analyze.
        threshold (float): The threshold value to use for classifying the sentiment.

    Returns:
        list: A list of sentiment labels, which are either 'positive', 'negative', or 'neutral'.
    """
    # Analyze the sentiment of the texts
    scores = batch_analyze_sentiment(texts)

    # Classify the sentiment based on the threshold
    labels = []
    for score in scores:
        if score > threshold:
            labels.append('positive')
        elif score < -threshold:
            labels.append('negative')
        else:
            labels.append('neutral')

    # Return the sentiment labels as a list
    return labels
