import pandas as pd
from bs4 import BeautifulSoup
import string
from unidecode import unidecode
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
    return unidecode(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def duplicate_row_by_tags(df: pd.DataFrame):
    new_rows = []

    # Parcourez chaque ligne du DataFrame
    for _, row in df.iterrows():
        body_clean = row['Body_clean']
        tags_clean = row['Tags_clean']

        # Parcourez chaque tag dans la liste de tags
        for tag in tags_clean:
            if tag == "":
                break
            # Créez une nouvelle ligne avec le même texte et le tag actuel
            new_row = {'Body_clean': body_clean, 'Tag': tag}
            new_rows.append(new_row)

    # Créez un nouveau DataFrame avec les nouvelles lignes
    new_df = pd.DataFrame(new_rows)
    return new_df

def most_frequent_tag_df(df: pd.DataFrame):
    tag_counts = df['Tag'].value_counts()

    # Sélectionnez les 25 tags les plus courants
    top_tags = tag_counts.head(25).index.tolist()

    # Filtrer le DataFrame en ne conservant que les lignes avec les tags les plus courants
    filtered_df = df[df['Tag'].isin(top_tags)]
    filtered_df["Tag"][filtered_df["Tag"]== "c#"] = "csharp"
    filtered_df["Tag"][filtered_df["Tag"]== "c++"] = "cplusplus"
    return filtered_df

# def clean_data(df: pd.DataFrame):
#     df = clean_tags(df)
#     df['Body_clean'] = df['Body'].apply(remove_stopwords)
#     df['Body_clean'] = df['Body'].apply(remove_accents)
#     return df.iloc[0][1]



def clean_data(df: pd.DataFrame):
    df['Body_clean'] = df['Body'].apply(clean_body)
    df['Body_clean'] = df['Body_clean'].apply(remove_stopwords).apply(remove_accents)
    df_clean = clean_tags(df)[["Body_clean", "Tags_clean"]]
    new_df = duplicate_row_by_tags(df_clean)
    filtered_df = most_frequent_tag_df(new_df)

    return filtered_df
