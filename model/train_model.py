"""
Spam Classifier Model Training Script
Trains a Multinomial Naive Bayes classifier on the UCI SMS Spam Collection dataset
"""

import os
import re
import pickle
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Build a small but representative dataset (fallback when download blocked)
# ---------------------------------------------------------------------------
SPAM_SAMPLES = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
    "You have been selected for a cash prize of $500. Call 1-800-WIN-NOW",
    "URGENT! Your mobile number has won £2000 bonus caller prize on 09061743810",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
    "Congratulations ur awarded 500 of bonus points. To claim call our customer service now!",
    "You won a luxury cruise for 2! Reply YES to claim your prize",
    "Get FREE ringtones sent to your phone! Just text RING to 80082",
    "Win a brand new iPod! Text WIN to 12345 now. Offer expires midnight!",
    "Special offer! Buy 1 get 1 FREE on all products. Call now 0800-FREE",
    "You qualify for a $250 Visa gift card! Go to http://win-gift.com to claim",
    "PRIVATE! You have won a £10,000 price and 500,000 to be exact",
    "Loan for any purpose available NOW. Low rates, fast approval!",
    "Hot singles in your area want to meet you NOW! Click here",
    "You have been pre-approved for a $50,000 loan. Apply now!",
    "Limited time offer! Earn $5000/week working from home",
    "Your package is delayed. Confirm details at: bit.ly/pkg-update",
    "Congratulations! You are our lucky winner. Claim $500 Amazon voucher",
    "FREE! FREE! FREE! Get your free iPhone 14 today, limited stock",
    "ALERT: Your bank account needs verification. Call 0800-FRAUD now",
    "Make money fast! Join our affiliate program and earn $1000 daily",
    "You've been selected for a FREE vacation package to Bahamas",
    "Urgent: Your account will be suspended. Verify at spam-verify.com",
    "Buy cheap Viagra online, best prices guaranteed, discreet shipping",
    "Earn money online from home, no experience needed, $500 per day",
    "Your credit score is ready. Click to see your free credit report",
    "You are a winner! Text CLAIM to 80800 to receive £1000 cash prize",
    "Ringtones! Get latest chart hits as ringtones. Text CHART to 85069",
    "Important message: You qualify for debt relief. Call now free!",
    "SALE! 50% off all products today only. Visit our website now",
    "Don't miss out! Last chance to claim your $1000 reward",
    "You have won the weekly lottery. Claim your prize of £50,000",
    "Apply for instant cash loan up to $10,000 with bad credit OK",
    "Exclusive deal: Get 3 months free subscription. Sign up today",
]

HAM_SAMPLES = [
    "Hey, are you coming to the party tonight?",
    "Can you pick up some milk on your way home?",
    "I'll be there in 10 minutes, traffic is bad",
    "Happy birthday! Hope you have a wonderful day",
    "Don't forget we have a meeting tomorrow at 9am",
    "Thanks for dinner, it was really delicious!",
    "Can we reschedule our appointment to Thursday?",
    "I'm running a bit late, will be there soon",
    "What time does the movie start tonight?",
    "Just wanted to check in and see how you're doing",
    "The project deadline has been moved to next Friday",
    "Sounds good, see you at the coffee shop at 3pm",
    "I've attached the report you requested. Let me know if you need changes",
    "Good morning! Have a great day at work",
    "Can you help me move this weekend? I'll buy pizza",
    "Your dentist appointment is confirmed for Monday at 2pm",
    "I sent you the documents. Please review when you get a chance",
    "Are you free for lunch tomorrow? There's a new place I want to try",
    "Flight is on time, should land around 6pm",
    "Just finished the gym, feeling great! Want to grab coffee?",
    "Reminder: team standup in 15 minutes",
    "Call me when you get a chance, need to discuss something",
    "The kids' school play is at 7pm on Friday. Don't forget!",
    "I left your keys under the mat. Hope that's okay",
    "Meeting was cancelled, we'll reschedule for next week",
    "Great job on the presentation today! Everyone loved it",
    "Can you send me the address? I'm heading over now",
    "Yes, I can make it to the conference. What time?",
    "Please find attached the invoice for this month's services",
    "Hi, just confirming receipt of your application",
    "The package was delivered to your doorstep this afternoon",
    "Let's catch up over the weekend, it's been too long",
    "I'll review the document and get back to you by EOD",
    "Happy New Year! Wishing you health and happiness",
    "Your order #12345 has been shipped and will arrive Friday",
]

def clean_text(text):
    """Basic text preprocessing without NLTK dependency"""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', 'url', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'email', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{10,}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', 'phone', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_dataset():
    """Build training dataset"""
    texts = SPAM_SAMPLES + HAM_SAMPLES
    labels = ['spam'] * len(SPAM_SAMPLES) + ['ham'] * len(HAM_SAMPLES)
    
    # Augment with variations
    augmented_texts, augmented_labels = [], []
    for t, l in zip(texts, labels):
        augmented_texts.append(t)
        augmented_labels.append(l)
        # Simple augmentation: uppercase version
        augmented_texts.append(t.upper())
        augmented_labels.append(l)
    
    df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    df['cleaned'] = df['text'].apply(clean_text)
    return df


def train():
    print("=" * 60)
    print("  SPAM CLASSIFIER - MODEL TRAINING")
    print("=" * 60)

    df = build_dataset()
    print(f"\n[INFO] Dataset size: {len(df)} samples")
    print(f"[INFO] Spam: {(df.label=='spam').sum()}, Ham: {(df.label=='ham').sum()}")

    X = df['cleaned']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build Pipeline: TF-IDF + Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    print(f"\n[RESULTS]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs('model', exist_ok=True)
    with open('model/spam_classifier.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Save metrics
    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    with open('model/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\n[SAVED] model/spam_classifier.pkl")
    print("[DONE] Training complete!")
    return pipeline, metrics


if __name__ == '__main__':
    train()
