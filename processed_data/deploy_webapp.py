# 23/CS/193

import streamlit as st
import joblib
import re
import string

# Load trained model and vectorizer
model = joblib.load("models/severity_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Streamlit UI
st.title("🛠️ Bug Severity Classifier")
st.write("Enter a bug report below to predict its severity level.")

user_input = st.text_area("Bug Description", height=150)

if st.button("Predict Severity"):
    if user_input.strip() == "":
        st.warning("Please enter a bug report.")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        st.success(f"🔍 Predicted Severity: **{pred.upper()}**")
