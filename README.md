 Sentiment Analysis using Machine Learning
 Project Description

This project is a **Sentiment Analysis System** built using Machine Learning techniques.
It classifies textual data into **positive** or **negative sentiment** using a Naive Bayes classifier.

The system processes input text, converts it into numerical features, and predicts sentiment based on learned patterns.

 Objectives

* To analyze user opinions from text data
* To classify sentiments into positive or negative
* To implement a basic NLP pipeline using Scikit-learn
* To evaluate model performance using accuracy and classification metrics

---

 Technologies Used

* **Python**
* **NumPy** – Numerical operations
* **Pandas** – Data handling
* **Scikit-learn** – Machine learning models and tools



 Project Workflow

1. Data Collection

A small dataset of sample text reviews is created manually:

* Positive reviews
* Negative reviews



2. Data Preprocessing

* Text data is cleaned using **stop word removal**
* Converted into numerical format using:

  * **CountVectorizer (Bag of Words model)**


 3. Train-Test Split

Dataset is divided into:

* Training Data (80%)
* Testing Data (20%)



4. Model Training

* Algorithm used: **Multinomial Naive Bayes**
* The model learns patterns from training data



5. Prediction

 Model predicts sentiment for unseen test data

---

6. Evaluation

Performance is evaluated using:

* **Accuracy Score**
* **Classification Report**

  * Precision
  * Recall
  * F1-score


 📊 Output Example

The model prints:

* Accuracy of prediction
* Detailed classification report



 Project Structure

```
sentiment-analysis/
│
├── sentiment.py      # Main Python file
├── README.md         # Project documentation
```

---

▶️ How to Run the Project

Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Step 2: Run the Code

```bash
python sentiment.py
```

---

## 🚀 Features

* Simple and easy-to-understand implementation
* Uses Machine Learning for sentiment classification
* Lightweight and beginner-friendly project
* Demonstrates basic NLP pipeline

---

## 🔮 Future Improvements

* Use large datasets (IMDB, Twitter, etc.)
* Add advanced NLP preprocessing
* Implement Deep Learning models (LSTM, BERT)
* Build a web interface using Streamlit or Flask
* Add real-time sentiment analysis

---

## ⚠️ Limitations

* Uses a very small dataset
* Limited accuracy due to less training data
* Basic text preprocessing only

---

## 👩‍💻 Author

**Pragya Pathak**

---

## ⭐ Conclusion

This project demonstrates how Machine Learning can be used for **text classification and sentiment analysis**.
It serves as a strong foundation for building more advanced NLP applications.
