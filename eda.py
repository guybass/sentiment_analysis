from typing import List, Dict
import string
import re
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


class WordCloudGenerator:
    """Generates a word cloud from a given text or file path."""

    def __init__(self, text: str = None, file_path: str = None) -> None:
        """
        Constructor for the WordCloudGenerator class.

        Args:
            text (str, optional): The text to be used for generating the word cloud.
            file_path (str, optional): The path to the text file to be used for generating the word cloud.
        """
        if text is not None:
            self.text = text
        elif file_path is not None:
            with open(file_path, 'r') as f:
                self.text = f.read()

    def clean_text(self, text: str=None) -> str:
        """
        Cleans the input text by removing special characters and stop words, and performing stemming and lemmatization.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        if text is None:
            text = self.text
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        porter_stemmer = PorterStemmer()

        # remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()

        # remove stopwords and lemmatize/stem the words
        words = word_tokenize(text)
        words = [porter_stemmer.stem(wordnet_lemmatizer.lemmatize(word, pos='v'))
                 for word in words if word not in stop_words]

        # return cleaned text as string
        return ' '.join(words)

    def get_word_counts(self, text: str=None) -> Dict[str, int]:
        """
        Counts the frequency of each word in the input text.

        Args:
            text (str): The text to be analyzed.

        Returns:
            dict: A dictionary containing the word counts.
        """
        if text is None:
            text = self.text
        words = self.clean_text(text).split()
        return dict(Counter(words))

    def get_top_n_words(self, text: str=None, n: int=1) -> List[str]:
        """
        Returns the n most frequent words in the input text.

        Args:
            text (str): The text to be analyzed.
            n (int): The number of top words to return.

        Returns:
            list: A list containing the n most frequent words.
        """
        if text is None:
            text = self.text
        word_counts = self.get_word_counts(text)
        return [word for word, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:n]]

    def get_word_frequencies(self, text: str=None) -> Dict[str, float]:
        """
        Calculates the frequency of each word in the input text.

        Args:
            text (str): The text to be analyzed.

        Returns:
            dict: A dictionary containing the word frequencies.
        """
        if text is None:
            text = self.text
        words = self.clean_text(text).split()
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        return {word: count / total_words for word, count in word_counts.items()}

    def get_word_cloud_image(self, text: str=None, mask_image: np.ndarray = None) -> Image.Image:
        """
        Generates a word cloud image from the input text.

        Args:
            text (str): The text to be analyzed.
            mask_image (np.ndarray, optional): A mask image to use for the word cloud.

        Returns:
            Image.Image: The generated word cloud image.
        """
        if text is None:
            text = self.text
        word_frequencies = self.get_word_frequencies(text)
        word_cloud = WordCloud(width=800, height=400, background_color='white', mask=mask_image).generate_from_frequencies(word_frequencies)
        return word_cloud.to_image()

    def plot_word_cloud(self, text: str=None, mask_image: np.ndarray = None) -> None:
        """
        Plots a word cloud image from the input text.

        Args:
            text (str): The text to be analyzed.
            mask_image (np.ndarray, optional): A mask image to use for the word cloud.
        """
        if text is None:
            text = self.text
        word_cloud_image = self.get_word_cloud_image(text, mask_image)
        plt.imshow(word_cloud_image)
        plt.axis('off')
        plt.show()
