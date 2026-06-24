# Spam Message Detection using Machine Learning

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier.

---

## Features

- Text preprocessing
  - Lowercasing
  - Removing punctuation and numbers
  - Stopword removal using NLTK

- TF-IDF Vectorization
  - Converts text into numerical feature vectors

- Stratified Train-Test Split
  - Preserves class distribution

- Model Training
  - Multinomial Naive Bayes

- Evaluation Metrics
  - Accuracy Score
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report

- Model Persistence
  - Saves trained model using Joblib
  - Saves TF-IDF vectorizer for future predictions

- Interactive Inference Mode
  - User can enter custom messages from terminal
  - Predicts Spam or Ham in real time

---

## Dataset

Dataset contains SMS messages labeled as:

- ham → Normal Message
- spam → Unwanted/Promotional Message

File:
spam.csv

---

## Project Structure

```text
SPAM_MESSAGE_DETECTION/
│
├── main_new.py
├── model_selection.py
├── spam.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into project folder:

```bash
cd SPAM_MESSAGE_DETECTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main_new.py
```

### First Run

* Dataset is loaded
* Text is cleaned
* TF-IDF vectorizer is trained
* Naive Bayes model is trained
* Evaluation metrics are displayed
* Model files are saved

### Future Runs

Saved model and vectorizer are loaded automatically.

Enter messages from terminal:

```text
Enter Message:
```

Example:

```text
Congratulations! You have won ₹50,000 cash.
```

Output:

```text
Prediction: spam
```

---

## Machine Learning Workflow

```text
Load Dataset
      ↓
Text Cleaning
      ↓
Train-Test Split
      ↓
TF-IDF Vectorization
      ↓
Naive Bayes Training
      ↓
Model Evaluation
      ↓
Save Model
      ↓
Real-Time Prediction
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Joblib

---


