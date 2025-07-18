# 📰 Fake News Detection using Machine Learning

This project aims to detect whether a news article is **real or fake** using Natural Language Processing (NLP) and Machine Learning techniques. The model is trained on labeled datasets of true and fake news and is deployed using **Streamlit** for interactive web-based predictions.

---

## 🔍 Project Overview

- **Domain**: NLP / Text Classification
- **Goal**: Classify news articles as "Real" or "Fake"
- **ML Model**: Logistic Regression
- **Deployment**: Streamlit (local and cloud support)
- **Dataset**: Combination of `True.csv` and `Fake.csv` files

---

## 📁 Project Structure

fake-news-detector/
│
├── data/
│ ├── True.csv
│ └── Fake.csv
│
├── train_model.py # Script to clean data, train model, and save it as model.pkl
├── model.pkl # Trained machine learning model
├── app.py # Streamlit app for fake news prediction
└── README.md # Project documentation


---

## ⚙️ Features

- Data cleaning and preprocessing using NLTK
- Text vectorization using `TfidfVectorizer`
- Model training using `LogisticRegression`
- Web interface for inputting custom news and checking if it's fake
- Easily deployable to the cloud using Streamlit

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/sandeshdesai/fake-news-detector.git
cd fake-news-detector

python train_model.py

streamlit run app.py
