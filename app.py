import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- LOAD NLTK ---------------- #

@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
load_nltk()
ps = PorterStemmer()

# ---------------- TEXT PREPROCESS ---------------- #

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    clean_words = []
    for word in words:
        if word.isalnum():
            if word not in stopwords.words("english"):
                clean_words.append(ps.stem(word))
    return " ".join(clean_words)


# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_assets():
    tfidf = pickle.load(open(".pkl files/vectorizer.pkl", "rb"))
    model = pickle.load(open(".pkl files/model.pkl", "rb"))
    return tfidf, model
tfidf, model = load_assets()

# ---------------- UI ---------------- #

st.title("🛡️ Spam SMS Classifier")

user_input = st.text_area(
    "Enter your SMS or Email message:",
    height=200
)

# ---------------- PREDICTION ---------------- #

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        clean_text = transform_text(user_input)
        vector_input = tfidf.transform([clean_text]).toarray()
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Safe Message")

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.write("Spam SMS Classification | Learning based Project")