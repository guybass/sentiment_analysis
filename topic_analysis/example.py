import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Define a pre-processing function to clean and prepare text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Return the preprocessed text as a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Define a function to extract features from the preprocessed text
def extract_features(text):
    # Mark negations in the text for sentiment analysis
    text = mark_negation(text)

    # Extract unigram features from the text
    unigram_feats = extract_unigram_feats(text.split())

    return unigram_feats

# Define a function to train a sentiment classifier on a labeled dataset
def train_sentiment_classifier(labeled_data):
    # Create a sentiment analyzer with NLTK
    sentiment_analyzer = SentimentAnalyzer()

    # Set the feature extractor
    sentiment_analyzer.set_feature_extractor(extract_features)

    # Split the labeled data into training and testing sets
    training_data = labeled_data[:int(len(labeled_data) * 0.8)]
    testing_data = labeled_data[int(len(labeled_data) * 0.8):]

    # Train a Naive Bayes classifier on the labeled data
    trainer = NaiveBayesClassifier.train
    classifier = sentiment_analyzer.train(trainer, training_data)

    # Evaluate the accuracy of the classifier on the testing data
    accuracy = sentiment_analyzer.evaluate(testing_data)

    # Return the classifier and accuracy
    return classifier, accuracy

# Define a function to classify the sentiment of new text using a trained classifier
def classify_sentiment(text, classifier):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Extract features from the preprocessed text
    features = extract_features(preprocessed_text)

    # Classify the sentiment of the text using the trained classifier
    label = classifier.classify(features)

    return label

# Example usage:
# Load a labeled dataset of news articles about the stock market and fashion industry
from nltk.corpus import movie_reviews
from nltk.corpus import reuters

stock_market_articles = [(preprocess_text(article.raw()), 'positive')
                         for article in reuters.fileids(categories=['money-fx', 'trade'])
                         if article.startswith('test')]

fashion_articles = [(preprocess_text(movie_reviews.raw(fileid)), 'negative')
                    for fileid in movie_reviews.fileids('neg')]

labeled_data = stock_market_articles + fashion_articles

# Train a sentiment classifier on the labeled dataset
classifier, accuracy = train_sentiment_classifier(labeled_data)

# Classify the sentiment of new text using the trained classifier
text = "The stock market experienced a major uptick today."
label = classify_sentiment(text, classifier)
print(label)
