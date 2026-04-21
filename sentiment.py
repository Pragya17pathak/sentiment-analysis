

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Loading Dataset

# Example dataset
data = pd.DataFrame({
    'text': [
        "I love this product, it's amazing!",
        "This is the worst purchase I ever made.",
        "Absolutely fantastic experience, highly recommend.",
        "Terrible quality, very disappointed.",
        "Great value for money, will buy again."
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive']
})


# PReprocess Text (NLP Basic)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train Naive Bayes Classifier
# Train model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Predict
y_pred = nb_model.predict(X_test_vec)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
