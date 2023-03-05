import nltk
from nltk.corpus import reuters
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Download the required nltk corpus
nltk.download('reuters')

# Load the Reuters dataset
reuters_data = []
for file_id in reuters.fileids():
    reuters_data.append(reuters.raw(file_id))


# Preprocess the text data
def preprocess_text(text):
    """
    Preprocess the text data.
    This function tokenizes the text, removes stop words, punctuation, and lemmatizes the tokens.
    :param text: str
    :return: list of str
    """

    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation
    tokens = [token for token in tokens if token.isalpha()]

    # Lemmatize the tokens
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Preprocess the entire Reuters dataset
reuters_tokens = [preprocess_text(text) for text in reuters_data]

# Create a dictionary from the tokenized documents
dictionary = Dictionary(reuters_tokens)

# Convert the tokenized documents into bag-of-words vectors
corpus = [dictionary.doc2bow(tokens) for tokens in reuters_tokens]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

# Print the top words for each topic
for topic_id in range(10):
    print(f"Topic {topic_id}: {lda_model.print_topic(topic_id)}")
