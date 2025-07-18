import streamlit as st
import pickle
import os

# Load model and vectorizer
model_path = os.path.join("app", "model.pkl")
vectorizer_path = os.path.join("app", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article text and we'll predict if it's **REAL** or **FAKE**.")

user_input = st.text_area("✍️ Paste the news content here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success("✅ This news is likely **REAL**.")
        else:
            st.error("🚫 This news is likely **FAKE**.")
