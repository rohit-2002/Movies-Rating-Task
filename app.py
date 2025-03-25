import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and features
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title="IMDb Movie Rating Predictor", layout="centered")

st.title("🎬 IMDb Movie Rating Predictor")
st.markdown("Predict the IMDb rating of an Indian movie based on its attributes.")

st.sidebar.header("Enter Movie Details")

year = st.sidebar.number_input("Release Year", min_value=1950, max_value=2030, value=2020)
duration = st.sidebar.number_input("Duration (minutes)", min_value=30, max_value=300, value=120)
votes = st.sidebar.number_input("Total Votes", min_value=0, step=100, value=1000)

director_avg = st.sidebar.slider("Director's Avg Rating", 1.0, 10.0, 6.0)
genre_avg = st.sidebar.slider("Genre Avg Rating", 1.0, 10.0, 6.5)

director_freq = st.sidebar.number_input("Director Frequency", min_value=0, value=10)
genre_freq = st.sidebar.number_input("Genre Frequency", min_value=0, value=20)
actor1_freq = st.sidebar.number_input("Actor 1 Frequency", min_value=0, value=15)
actor2_freq = st.sidebar.number_input("Actor 2 Frequency", min_value=0, value=10)
actor3_freq = st.sidebar.number_input("Actor 3 Frequency", min_value=0, value=5)

if st.button("Predict Rating"):
    input_data = pd.DataFrame([[
        year, duration, votes,
        director_avg, genre_avg,
        director_freq, genre_freq,
        actor1_freq, actor2_freq, actor3_freq
    ]], columns=features)

    rating_pred = model.predict(input_data)[0]
    st.success(f"🎯 Predicted IMDb Rating: **{round(rating_pred, 2)} / 10**")
