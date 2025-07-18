# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# STEP 1: Load and Label Data
# -----------------------------
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1  # 1 = REAL
fake_df["label"] = 0  # 0 = FAKE

# Combine datasets
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# STEP 2: Create Input Text Column
# -----------------------------
# If both "title" and "text" exist, combine them; else just use "text"
if "title" in combined_df.columns and "text" in combined_df.columns:
    combined_df["content"] = combined_df["title"].fillna("") + " " + combined_df["text"].fillna("")
else:
    combined_df["content"] = combined_df["text"].fillna("")

X = combined_df["content"]
y = combined_df["label"]

# -----------------------------
# STEP 3: Vectorize Text
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# -----------------------------
# STEP 4: Train the Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# STEP 6: Save the Model and Vectorizer
# -----------------------------
with open("app/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("app/vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved in 'app/' folder.")
