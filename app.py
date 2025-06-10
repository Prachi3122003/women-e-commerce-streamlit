import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Page Configuration
st.set_page_config(page_title="🧥 Clothing Review Sentiment Analysis", layout="centered")

st.title("👗 Clothing E-commerce Sentiment Analyzer")
st.write("Analyze customer reviews and predict sentiment (positive/negative)")

# Load dataset directly
df = pd.read_csv("dataset.csv")  # Ensure this file is in the same folder as app.py

# Show raw data
st.subheader("📊 Raw Data")
st.dataframe(df.head())

# Drop missing reviews
df = df.dropna(subset=['Review Text'])

# Create sentiment label
df['Sentiment'] = df['Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')

# Display sentiment distribution
st.subheader("📈 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Sentiment', palette='Set2', ax=ax)
st.pyplot(fig)

# Train Model
st.subheader("🧠 Model Training")

X = df['Review Text']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report
st.write("🔍 **Classification Report**")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
st.write("📉 **Confusion Matrix**")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)
st.pyplot(fig)

# Try It Yourself
st.subheader("📝 Try It Yourself")
user_review = st.text_area("Enter a clothing review to predict sentiment:")
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        pred = model.predict([user_review])[0]
        st.success(f"Predicted Sentiment: **{pred}**")
