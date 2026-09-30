\# Phishing Email Detection Model



A machine learning project that classifies email messages as \*\*Phishing\*\* or \*\*Safe\*\* using Natural Language Processing (NLP).



\## Project Overview



This project demonstrates a basic phishing email detection system using:



\- TF-IDF text feature extraction

\- Logistic Regression classification

\- Training and testing dataset

\- Accuracy evaluation

\- Classification report

\- Confusion matrix

\- Prediction of new email messages



The project was developed as part of a cybersecurity internship mini project.



\## Features



\- Classifies emails as Phishing or Safe.

\- Converts email text into numerical features using TF-IDF.

\- Uses Logistic Regression for classification.

\- Splits the dataset into training and testing sets.

\- Calculates model accuracy.

\- Generates a classification report.

\- Generates and saves a confusion matrix.

\- Tests the model on new email examples.

\- Generates a text-based model report.



\## Technologies Used



\- Python 3

\- Pandas

\- Scikit-learn

\- Matplotlib

\- TF-IDF

\- Logistic Regression



\## Dataset



The project uses a small educational dataset containing:



\- 40 email examples

\- 20 Phishing emails

\- 20 Safe emails



The dataset is included directly in the Python program for demonstration purposes.



\## Model Training



The dataset is divided into:



\- 75% training data

\- 25% testing data



The model uses:



```text

TF-IDF Vectorizer

&#x20;       ↓

Logistic Regression

&#x20;       ↓

Phishing / Safe

