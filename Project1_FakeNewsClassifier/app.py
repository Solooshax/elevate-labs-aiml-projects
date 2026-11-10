import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 News Article Classification (Fake/Real)")

# Reload model directly without stale cache
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Please run `python train.py` first!")
    st.stop()

user_input = st.text_area("Enter Headline or Article Content:", height=130)

if st.button("Classify Text"):
    if not user_input.strip():
        st.warning("Please enter some text to test.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = probs[classes.index(prediction)] * 100

        st.write("---")
        if prediction == "REAL":
            st.success(f"✅ **Verdict: REAL NEWS** (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"🚨 **Verdict: FAKE NEWS** (Confidence: {confidence:.1f}%)")