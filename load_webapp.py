# 23/CS/193

import streamlit as st
import joblib
import re
import string
import pandas as pd

# Load model and vectorizer
model = joblib.load("models/severity_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI setup
st.set_page_config(page_title="Bug Severity Classifier", layout="centered")

st.title("🛠️ Bug Severity Classifier")
st.write("Enter a bug report or upload a file to classify severity.")

# Sidebar theme tip
st.sidebar.markdown("🌓 **Theme Tip:** Right-click → Appearance → Switch theme")

# Manual Input
st.subheader("📝 Manual Bug Report")
user_input = st.text_area("Describe the bug here", height=150)

if st.button("Predict Severity"):
    if user_input.strip() == "":
        st.warning("Please enter a bug report.")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        st.success(f"🔍 Predicted Severity: **{pred.upper()}**")

# File Upload Section
st.markdown("---")
st.subheader("📁 Upload CSV File of Bug Reports")

uploaded_file = st.file_uploader("Upload a CSV file with a column named 'bug_description'", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'bug_description' not in df.columns:
            st.error("❌ CSV must have a 'bug_description' column.")
        else:
            df['cleaned'] = df['bug_description'].apply(clean_text)
            vecs = vectorizer.transform(df['cleaned'])
            df['Predicted Severity'] = model.predict(vecs)

            st.success("✅ Prediction completed.")
            st.write(df[['bug_description', 'Predicted Severity']])

            # Optionally allow download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", data=csv, file_name="predicted_severities.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

# Feedback Section
st.markdown("---")
st.subheader("📝 Feedback")
feedback = st.text_area("Was the prediction helpful? Any suggestions?", key="feedback")

if st.button("Submit Feedback"):
    if feedback.strip():
        st.success("✅ Thanks for your feedback!")
        # Optional: Save to log file or database
    else:
        st.warning("Please write something before submitting.")
