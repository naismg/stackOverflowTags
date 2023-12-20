import pandas as pd
from bs4 import BeautifulSoup
import string
import unidecode
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def monitoring(path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_body(body):
    return BeautifulSoup(body, 'html.parser').get_text(separator=' ').lower().translate(str.maketrans('', '', string.punctuation))


def clean_tags(df):
    df['Body'] = df['Body'].apply(clean_body)
    df['Tags_clean'] = df['Tags'].str.replace('<', '', regex=True)
    df['Tags_clean'] = df['Tags_clean'].str.replace('>', ',', regex=True).str.split(",")
    return df


def remove_accents(text):
    return unidecode.unidecode(text)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def clean_data(df: pd.DataFrame, destination):
    df['Body_clean'] = df['Body'].apply(clean_body)
    df['Body_clean'] = df['Body_clean'].apply(remove_stopwords)
    df['Body_clean'] = df['Body_clean'].apply(remove_accents)
    df = clean_tags(df)
    df.to_csv(destination, index=False)


path = './data/raw/data.csv'
destination = './data/clean/data_clean.csv'

df = monitoring(path)
clean_data(df, destination)
