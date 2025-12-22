import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Student Math Score Predictor", layout="centered")

st.title("Student Math Score Predictor")
st.write(
    "This interactive dashboard demonstrates the trained machine learning model "
    "used in the thesis project to predict student math performance."
)

st.sidebar.header("Student Information")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
ethnicity = st.sidebar.selectbox(
    "Race / Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)
parent_edu = st.sidebar.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

reading = st.sidebar.slider("Reading Score", 0, 100, 70)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)

input_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [ethnicity],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [prep],
    "reading score": [reading],
    "writing score": [writing]
})

if st.button("Predict Math Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Math Score: **{prediction:.1f}**")
