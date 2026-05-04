
# 1. Import Libraries


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# 2. Load Dataset


cols = ["target","id","date","flag","user","text"]

df = pd.read_csv("training.1600000.processed.noemoticon.csv",
                 encoding="latin-1", names=cols)

# Balance dataset
df_neg = df[df['target'] == 0].sample(50000, random_state=42)
df_pos = df[df['target'] == 4].sample(50000, random_state=42)

df = pd.concat([df_neg, df_pos])


# 3. Preprocessing

df['target'] = df['target'].replace(4,1)
df = df[['target','text']]
df['text'] = df['text'].str.lower().str.replace(r'[^a-z\s]', ' ', regex=True)



# 4. Feature Extraction

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X = tfidf.fit_transform(df['text'])
y = df['target']



# 5. Cross Validation


nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)

nb_cv = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
lr_cv = cross_val_score(lr, X, y, cv=5, scoring='accuracy')

print("Naive Bayes CV Accuracy:", nb_cv.mean())
print("Logistic Regression CV Accuracy:", lr_cv.mean())



# 6. Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# 7. Train Models

nb.fit(X_train, y_train)
lr.fit(X_train, y_train)



# 8. Predictions


nb_pred = nb.predict(X_test)
lr_pred = lr.predict(X_test)



# 9. Evaluation


nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nNaive Bayes Accuracy:", nb_acc)
print("Logistic Regression Accuracy:", lr_acc)

print("\nNaive Bayes Report:\n", classification_report(y_test, nb_pred))
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))

cm = confusion_matrix(y_test, nb_pred)
print("\nConfusion Matrix:\n", cm)


# 10. BAR GRAPH (Accuracy)



models = ['Naive Bayes', 'Logistic Regression']
accuracy = [nb_acc, lr_acc]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()



# 11. PIE CHART (Class Distribution)

counts = df['target'].value_counts()

plt.figure()
plt.pie(counts, labels=['Negative','Positive'], autopct='%1.1f%%')
plt.title("Class Distribution")
plt.show()



# 12. HISTOGRAM (Prediction Distribution)

plt.figure()
plt.hist(nb_pred, bins=2)
plt.title("Prediction Distribution (Naive Bayes)")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()



# 13. SCATTER (Precision vs Recall)


# Extract values manually (from report)

precision = [0.76, 0.77, 0.79, 0.78]
recall = [0.78, 0.75, 0.78, 0.80]

plt.figure()
plt.scatter(precision, recall)
plt.title("Precision vs Recall")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.show()



# 14. Custom Prediction


def predict_sentiment(text):
    text = text.lower()
    vector = tfidf.transform([text])
    result = nb.predict(vector)[0]
    return "Positive 😊" if result == 1 else "Negative 😞"

print("\nCustom Prediction:")
print(predict_sentiment("I love this product"))
print(predict_sentiment("Worst experience ever"))