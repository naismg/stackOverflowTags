import pandas as pd
from bs4 import BeautifulSoup
import string
import unidecode
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
from supabase import create_client, Client
import os

load_dotenv()

URL = os.getenv('URL')
KEY = os.getenv('KEY')

nltk.download('stopwords')

def monitoring() -> pd.DataFrame:
    supabase_client = create_client(URL, KEY)
    response = supabase_client.table('stack_tags').select("*").execute()
    data = response.data
    return pd.DataFrame(data)


def clean_body(df):
    return BeautifulSoup(
        df, 'html.parser'
        ).get_text(separator=' '
                   ).lower(
                       ).translate(str.maketrans('', '', string.punctuation))

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

def clean_data(df: pd.DataFrame):
    df = clean_tags(df)
    df['Body_clean'] = df['Body'].apply(remove_stopwords)
    df['Body_clean'] = df['Body'].apply(remove_accents)
    return df.iloc[0][1]
