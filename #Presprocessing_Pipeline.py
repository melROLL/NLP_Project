#Presprocessing Pipeline

# Standard Data Handling and Analysis Libraries
import numpy as np
import pandas as pd

# Regular Expressions for Text Processing
import re

# Text Preprocessing and NLP Libraries
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Sentiment Analysis and Text Sentiment Scoring
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text Data Exploration and Analysis
from collections import Counter

# Word Embedding and Machine Learning Libraries
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Advanced Natural Language Processing (NLP) Tools
import spacy

# Disable pandas warnings
pd.options.mode.chained_assignment = None

# Additional Libraries for a Text Classification Example
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function to remove punctuation, stopwords, and perform stemming
def preprocess_text(text, remove_punctuation=True, remove_stopwords=True, perform_stemming=True):
    # Remove punctuation from the text (if enabled)
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove stopwords from the text (if enabled)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = " ".join(words)
    
    # Perform stemming on the text (if enabled)
    if perform_stemming:
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        text = " ".join(stemmed_words)
    
    return text

# Apply the preprocess_text function to a DataFrame's 'text' column
def clean_and_print(df, remove_punctuation=True, remove_stopwords=True, perform_stemming=True):
    df["text"] = df["text"].apply(lambda text: preprocess_text(text, remove_punctuation, remove_stopwords, perform_stemming))
    print("After preprocessing:")
    print(df.head().text.values)
    return df.head()

# Load the first CSV file containing true news data
Tr = "Data\True.csv"
dfT = pd.read_csv(Tr)

# Load the second CSV file containing fake news data
Fa = "Data\Fake.csv"
dfF = pd.read_csv(Fa)

# Preprocess text in the DataFrames, removing punctuation, stopwords, and performing stemming
clean_and_print(dfT, remove_punctuation=True, remove_stopwords=True, perform_stemming=True)
clean_and_print(dfF, remove_punctuation=True, remove_stopwords=True, perform_stemming=True)

