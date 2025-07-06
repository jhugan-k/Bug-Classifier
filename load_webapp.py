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

# Page config
st.set_page_config(page_title="Bug Severity Classifier", layout="centered")
st.title("🛠️ Bug Severity Classifier")
st.write("Choose how you want to classify bug severity:")


# Mode selection
mode = st.radio("Select Mode", ["🔍 Predict from text", "📁 Upload CSV file", "📝 Give Feedback"])

# 🔍 Predict from manual text
if mode == "🔍 Predict from text":
    st.subheader("Manual Bug Report")
    user_input = st.text_area("Describe the bug here", height=150)
    if st.button("Predict Severity"):
        if user_input.strip() == "":
            st.warning("Please enter a bug report.")
        else:
            clean = clean_text(user_input)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]
            st.success(f"🔍 Predicted Severity: **{pred.upper()}**")

# 📁 Predict from uploaded CSV
elif mode == "📁 Upload CSV file":
    st.subheader("Upload a CSV with a 'bug_description' column")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'bug_description' not in df.columns:
                st.error("❌ The CSV must contain a column named 'bug_description'.")
            else:
                df['cleaned'] = df['bug_description'].apply(clean_text)
                vecs = vectorizer.transform(df['cleaned'])
                df['Predicted Severity'] = model.predict(vecs)

                st.success("✅ Prediction complete.")
                st.write(df[['bug_description', 'Predicted Severity']])

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", data=csv, file_name="predicted_severities.csv", mime='text/csv')
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# 📝 Feedback form
elif mode == "📝 Give Feedback":
    st.subheader("Your Feedback")
    feedback = st.text_area("How was your experience? Suggestions?", key="feedback")
    if st.button("Submit Feedback"):
        if feedback.strip():
            st.success("✅ Thank you! Your feedback has been received.")
            # Optional: append to a log file or database
        else:
            st.warning("Please write something before submitting.")

