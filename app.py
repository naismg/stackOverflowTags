import streamlit as st
import pandas as pd
import os
from fastapi import FastAPI
import uvicorn
import requests


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


    </style>
    """,
            unsafe_allow_html=True)


app = FastAPI()

user_input = st.text_input("Entrez du texte ici")

response = requests.post('https://stack-tags.onrender.com/predict', json={'text': user_input})

prediction = response.json()['prediction']

with st.container():
   st.empty()

st.write(f'La prédiction pour "{user_input}" est: {prediction}')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
