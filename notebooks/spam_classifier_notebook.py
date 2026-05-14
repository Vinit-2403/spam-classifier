# Spam Classifier - Jupyter Notebook
# Run these cells in order in Jupyter or Google Colab

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# !pip install scikit-learn pandas numpy matplotlib seaborn nltk

# ============================================================
# CELL 2 — Imports
# ============================================================
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.pipeline import Pipeline

print("All imports successful!")

# ============================================================
# CELL 3 — Dataset (Extended)
# ============================================================
# In production: use the UCI SMS Spam Collection dataset
# Download: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','text'])

# For demonstration, we build a representative dataset
SPAM = [
    "WINNER!! You have been selected to receive £900 prize! Call 09061701461",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim",
    "You have been selected for a cash prize of $500. Call 1-800-WIN",
    "URGENT! Your mobile number has won £2000 bonus caller prize",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11",
    "FREE ringtones! Text RING to 80082. Terms apply.",
    "Win a brand new iPhone! Text WIN to 12345. Offer expires midnight!",
    "PRIVATE! You have won £10,000 price. Call now to claim.",
    "Hot singles in your area. Click here. Limited time offer!",
    "You qualify for a $250 Visa card! Go to http://win-gift.com",
    "Loan for any purpose. LOW rates. Fast approval. Apply now!",
    "Make money fast! Join our affiliate program. $1000 daily.",
    "Your package is delayed. Confirm details: bit.ly/pkg-update",
    "Buy cheap Viagra online. Best prices. Discreet shipping.",
]

HAM = [
    "Hey, are you coming to the party tonight?",
    "Can you pick up some milk on your way home?",
    "I'll be there in 10 minutes, traffic is bad",
    "Don't forget we have a meeting tomorrow at 9am",
    "Thanks for dinner, it was really delicious!",
    "Can we reschedule our appointment to Thursday?",
    "What time does the movie start tonight?",
    "Just wanted to check in and see how you're doing",
    "The project deadline has been moved to next Friday",
    "Your dentist appointment is confirmed for Monday at 2pm",
    "I sent you the documents. Please review when you get a chance",
    "Are you free for lunch tomorrow? New place I want to try",
    "Flight is on time, should land around 6pm",
    "Reminder: team standup in 15 minutes",
    "Great job on the presentation today! Everyone loved it",
]

texts = SPAM + HAM
labels = ['spam'] * len(SPAM) + ['ham'] * len(HAM)
df = pd.DataFrame({'text': texts, 'label': labels})

print("Dataset shape:", df.shape)
print(df['label'].value_counts())
df.head()

# ============================================================
# CELL 4 — Text Preprocessing
# ============================================================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'\b\d{10,}\b', 'phone', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['text'].apply(preprocess)
print("Sample cleaned text:")
print(df[['text','cleaned']].head(3).to_string())

# ============================================================
# CELL 5 — Train/Test Split
# ============================================================
X = df['cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================================
# CELL 6 — Build and Train Pipeline
# ============================================================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== MODEL RESULTS ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred, pos_label='spam'):.4f}")
print()
print(classification_report(y_test, y_pred))

# ============================================================
# CELL 7 — Confusion Matrix Visualization
# ============================================================
cm = confusion_matrix(y_test, y_pred, labels=['spam','ham'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['spam','ham'], yticklabels=['spam','ham'])
plt.title('Confusion Matrix — Spam Classifier', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# CELL 8 — Cross Validation
# ============================================================
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
print(f"5-Fold CV F1 scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# CELL 9 — Compare Multiple Models
# ============================================================
models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(max_iter=1000),
}

results = []
for name, clf in models.items():
    p = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), ('clf', clf)])
    p.fit(X_train, y_train)
    yp = p.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp, pos_label='spam'),
        'Recall': recall_score(y_test, yp, pos_label='spam'),
        'F1': f1_score(y_test, yp, pos_label='spam'),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ============================================================
# CELL 10 — Save Model
# ============================================================
import os
os.makedirs('model', exist_ok=True)
with open('model/spam_classifier.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved to model/spam_classifier.pkl")

# ============================================================
# CELL 11 — Test Predictions
# ============================================================
test_messages = [
    "Hey, want to grab coffee tomorrow morning?",
    "CONGRATULATIONS! You've won a FREE iPhone! Click now to claim!",
    "Meeting is rescheduled to 3pm. Please confirm attendance.",
    "URGENT: Your account will be closed. Verify at: bit.ly/acct",
]

for msg in test_messages:
    pred = pipeline.predict([preprocess(msg)])[0]
    proba = pipeline.predict_proba([preprocess(msg)])[0]
    spam_p = proba[list(pipeline.classes_).index('spam')]
    print(f"[{pred.upper():<4}] {spam_p:.0%} spam | {msg[:60]}")
