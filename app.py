import streamlit as st
import pandas as pd
import os
from fastapi import FastAPI
import uvicorn
import requests
import mlflow
import plotly.express as px

# generate_background_image()
st.set_page_config(layout="wide")


app = FastAPI()

col, col1 = st.columns(2)

col.empty()
col1.empty()

col3, col4, col5 = st.columns(3)

col.empty()
col1.empty()

user_input = col4.text_input("")

response = requests.post('https://stack-tags.onrender.com/predict', json={'text': user_input})

prediction = response.json()['prediction']

col6, col7, col8 = st.columns(3)

col7.write(f'La prédiction pour "{user_input}" est: {prediction}')

response2 = requests.get('https://stack-tags.onrender.com/search_runs')

prediction2 = response2.json()['runs']

df = pd.DataFrame(prediction2)

y_options = df.columns.tolist()

y_choice = st.selectbox('Choisissez une colonne pour l\'axe y', y_options)

fig = px.scatter(
    df,
    x="start_time",
    y=y_choice,
    color="status",
    hover_data=["run_id", "experiment_id", y_choice],
    labels={"start_time": "Start Time", "end_time": "End Time"},
    title="Run Status and Accuracy Over Time",
)

fig.update_layout(yaxis=dict(range=[-1, 1]))

st.plotly_chart(fig)



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
