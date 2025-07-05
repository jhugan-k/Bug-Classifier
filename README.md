# Bug Severity Classifier 

This is a machine learning-based app that predicts the **severity of software bugs** from plain text bug reports. You can type in or paste any bug description, and the model will classify it as one of the following:

- 🔴 Critical  
- 🟠 High  
- 🟡 Medium  
- 🟢 Low

The app is built using Python, trained on real-world bug data, and deployed using Streamlit.

---

##  Features

- Classifies severity of any bug description using NLP + ML
- Clean text processing with lemmatization and stopword removal
- Trained on thousands of labeled bug reports
- Lightweight web interface (no sign-in, works instantly)
- Open-source and easy to extend

---

##  How It Works

- Text is cleaned and converted into numeric features using TF-IDF
- A logistic regression model (from scikit-learn) predicts severity
- The app runs fully in your browser via Streamlit

---

##  Tech Stack

- Python 3.11  
- pandas  
- scikit-learn  
- joblib  
- Streamlit  

---

##  Try it Online

Try the live app here:  
> [https://bug-classifier.streamlit.app](https://bug-classifier.streamlit.app) *(replace with your link)*

---

##  Future Plans

- Add bug **category** prediction (e.g., UI Bug, Security Bug, etc.)
- Add CSV file upload to classify bugs in bulk
- Show prediction confidence scores
- Collect user feedback on predictions

---

##  Feedback

Found a bug in the bug classifier?  
Raise an issue or message me — happy to improve it!

>jkartikey.official@gmail.com
>https://www.linkedin.com/in/jhugan-kartikey-068a60315/


