import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, 'data', 'raw', 'SMSSpamCollection.csv')
    df = pd.read_csv(file_path, encoding='latin1')
    # Convert labels to binary
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis =1)
    return df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    final_text = ' '.join(tokens)
    return final_text

def prepare_dataset(df):
    df['clean_message'] = df['Message'].apply(preprocess_text)
    return df