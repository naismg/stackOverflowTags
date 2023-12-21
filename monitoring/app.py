import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import os

# generate_background_image()
st.set_page_config(layout="wide")

background_image = 'url("https://www.silicon.fr/wp-content/uploads/2019/08/stack_overflow.jpg")'
# CSS pour définir l'image de fond
st.markdown(f"""
    <style>
        .stApp {{
    background-image: {background_image};
    background-repeat: no-repeat;
    background-color: white;
    background-position: top;
    height: 50vh; /* Ajustez cette valeur pour changer la hauteur de l'image */
}}

    .sidebar .sidebar-content {{
        background: #262730;
    }}
    .Widget {{
        color: white;
    }}
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stFileUploader > div > input {{
        background-color: #555e6f;
        color: white;
    }}
    .stTextInput > div > div > input::placeholder, .stTextArea > div > div > textarea::placeholder {{
        color: white;
    }}
    .stCheckbox label, .stRadio label {{
        color: white;
    }}
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: black;
    }}
    .stSelectbox div[role="button"] {{
        color: white;
    }}
    </style>
    """,
            unsafe_allow_html=True)

load_dotenv()

key = os.getenv("KEY")
url = os.getenv("URL")

supabase: Client = create_client(url, key)
response = supabase.table('stack_tags').select("*").execute()
stack = pd.DataFrame(response.data)