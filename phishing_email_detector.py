import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# PHISHING EMAIL DETECTION MODEL
# TF-IDF + Logistic Regression
# ============================================================


# ------------------------------------------------------------
# 1. Create a small educational email dataset
# ------------------------------------------------------------

data = {
    "email": [
        # Phishing emails
        "Urgent! Your bank account has been suspended. Click here immediately to verify your account.",
        "Congratulations! You won a $1000 prize. Click the link to claim your reward now.",
        "Your password will expire today. Verify your account immediately using the link below.",
        "Security alert! We detected unusual activity. Confirm your login information now.",
        "Your package could not be delivered. Pay the delivery fee using this link.",
        "You have received a refund. Click here and enter your bank details to receive the money.",
        "Your email account will be deleted. Confirm your password immediately.",
        "FINAL WARNING! Your account will be locked unless you verify your identity now.",
        "You have won a lottery prize. Send your personal details to claim the money.",
        "Click this link to update your banking information and avoid account suspension.",
        "We detected suspicious login activity. Sign in now to secure your account.",
        "Your payment failed. Update your card details immediately using the attached link.",
        "Important security notice! Your account requires immediate verification.",
        "Claim your free gift card today by clicking the link and entering your details.",
        "Your online banking access has been restricted. Verify your account now.",
        "You are eligible for an exclusive cash reward. Click here to claim it.",
        "Your account has been compromised. Send your password to our security team.",
        "Urgent invoice problem! Open the attached file and confirm your payment information.",
        "Your subscription will be cancelled today unless you update your payment details.",
        "Congratulations! You have been selected for a special reward. Claim it immediately.",

        # Safe / legitimate emails
        "Your monthly bank statement is now available in your secure online banking account.",
        "Thank you for attending today's meeting. The presentation is attached for your reference.",
        "Your order has shipped and can be tracked from your account on our official website.",
        "Reminder: Your appointment with the clinic is scheduled for tomorrow at 10 AM.",
        "Your electricity bill is ready to view in your customer account.",
        "The project meeting has been moved to Friday at 3 PM.",
        "Thank you for your purchase. Your receipt is available in your account.",
        "Your university timetable has been updated. Please check the student portal.",
        "The library book you borrowed is due next Monday.",
        "Your application has been received successfully. We will contact you with further information.",
        "Your flight booking confirmation is available in your travel account.",
        "The team meeting notes have been uploaded to the shared company folder.",
        "Your monthly subscription payment was processed successfully.",
        "Please find the meeting agenda attached for tomorrow's discussion.",
        "Your package has been delivered to the address provided during checkout.",
        "The examination schedule is available on the official university portal.",
        "Thank you for contacting customer support. Your ticket has been created.",
        "Your employee training session is scheduled for Wednesday morning.",
        "Your insurance policy document is available in your customer portal.",
        "The requested report has been completed and is attached to this email."
    ],

    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",

        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe"
    ]
}


df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. Display dataset information
# ------------------------------------------------------------

print("=" * 65)
print("           PHISHING EMAIL DETECTION MODEL")
print("=" * 65)

print("\nDataset Information")
print("-" * 65)
print(f"Total emails: {len(df)}")
print(f"Phishing emails: {(df['label'] == 'Phishing').sum()}")
print(f"Safe emails: {(df['label'] == 'Safe').sum()}")


# ------------------------------------------------------------
# 3. Split dataset into training and testing data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)


# ------------------------------------------------------------
# 4. Convert email text into numerical TF-IDF features
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ------------------------------------------------------------
# 5. Train Logistic Regression model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# ------------------------------------------------------------
# 6. Predict test emails
# ------------------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# ------------------------------------------------------------
# 7. Calculate accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 65)
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 8. Display classification report
# ------------------------------------------------------------

print("\nClassification Report")
print("-" * 65)
print(classification_report(y_test, y_pred, zero_division=0))


# ------------------------------------------------------------
# 9. Generate confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Safe", "Phishing"]
)

print("Confusion Matrix")
print("-" * 65)
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

disp.plot()
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. Test the model with new emails
# ------------------------------------------------------------

test_emails = [
    "Urgent! Verify your bank password immediately by clicking this link.",
    "The project meeting is scheduled for tomorrow at 11 AM.",
    "Congratulations! You won a cash prize. Claim your reward now.",
    "Your university examination timetable is available on the student portal."
]

test_features = vectorizer.transform(test_emails)
predictions = model.predict(test_features)


print("\nNew Email Predictions")
print("-" * 65)

for email, prediction in zip(test_emails, predictions):
    print(f"\nEmail: {email}")
    print(f"Prediction: {prediction}")


# ------------------------------------------------------------
# 11. Save the model results to a text report
# ------------------------------------------------------------

with open("model_report.txt", "w", encoding="utf-8") as report:

    report.write("=" * 65 + "\n")
    report.write("           PHISHING EMAIL DETECTION REPORT\n")
    report.write("=" * 65 + "\n\n")

    report.write(f"Dataset size: {len(df)} emails\n")
    report.write(f"Training samples: {len(X_train)}\n")
    report.write(f"Testing samples: {len(X_test)}\n\n")

    report.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

    report.write("Confusion Matrix\n")
    report.write(str(cm))
    report.write("\n\n")

    report.write("New Email Predictions\n")
    report.write("-" * 65 + "\n")

    for email, prediction in zip(test_emails, predictions):
        report.write(f"Email: {email}\n")
        report.write(f"Prediction: {prediction}\n\n")


print("\nModel report saved as: model_report.txt")
print("Confusion matrix saved as: confusion_matrix.png")

print("\nTraining and testing completed successfully.")
print("=" * 65)