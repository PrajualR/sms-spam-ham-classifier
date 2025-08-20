import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = ['punkt','punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"Resource '{resource}' already downloaded")
        except LookupError:
            print(f"Downloading '{resource}'...")
            nltk.download(resource, quiet=True)

# Call this function at import time
download_nltk_resources()

def load_data():
    """Load and merge both datasets"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path_1 = os.path.join(base_path, 'data', 'raw', 'SMSSpamCollection.csv')
    file_path_2 = os.path.join(base_path, 'data', 'raw', 'Dataset_2025.csv')

    df1 = pd.read_csv(file_path_1, encoding='latin1')
    df2 = pd.read_csv(file_path_2, encoding='latin1')

    # Convert labels to binary
    df1['Label'] = df1['Label'].map({'ham': 0, 'spam': 1})
    df2['Label'] = df2['Label'].map({'ham': 0, 'spam': 1, 'Smishing': 1})

    # Drop unnecessary columns from df1
    if set(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']).issubset(df1.columns):
        df1 = df1.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

    # Merge and shuffle
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.dropna()
    return df


def preprocess_text(text):
    text = text.lower()

    # Replace URLs with token
    text = re.sub(r'http\S+|www\S+|https\S+', 'URLTOKEN', text)
    # Replace numbers with token
    text = re.sub(r'\d+', 'NUMTOKEN', text)
    # Remove special characters (keep only words, numbers, tokens, whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def prepare_dataset(df):
    df['clean_message'] = df['Message'].apply(preprocess_text)
    return df