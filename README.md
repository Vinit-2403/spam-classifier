# 🛡️ SpamShield — Email & SMS Spam Classifier

> **End-to-End Machine Learning Project** | Naive Bayes + TF-IDF | Flask REST API | Heroku Deployment

---

## 📌 Project Overview

SpamShield is a complete **Email and SMS Spam Classification** system built from scratch using Machine Learning. It features:

- ✅ ML Pipeline: TF-IDF Vectorizer + Multinomial Naive Bayes
- ✅ REST API backend built with Flask
- ✅ Modern, responsive frontend UI
- ✅ Single & Batch message classification
- ✅ Spam indicator detection
- ✅ Probability scores with confidence levels
- ✅ Heroku deployment ready

---

## 🏗️ Project Structure

```
spam-classifier/
├── app.py                   # Flask application (backend)
├── requirements.txt         # Python dependencies
├── Procfile                 # Heroku process file
├── runtime.txt              # Python runtime version
├── .gitignore
├── model/
│   ├── train_model.py       # Model training script
│   ├── spam_classifier.pkl  # Trained model (generated)
│   └── metrics.pkl          # Model metrics (generated)
├── templates/
│   └── index.html           # Frontend UI
└── notebooks/
    └── spam_classifier_notebook.py   # EDA + training notebook
```

---

## 🧠 ML Pipeline

```
Raw Text
   ↓
Text Preprocessing
   → Lowercase
   → URL normalization (url)
   → Email normalization (email)
   → Phone normalization (phone)
   → Remove special characters
   ↓
TF-IDF Vectorizer
   → max_features=5000
   → ngram_range=(1, 2)
   → sublinear_tf=True
   ↓
Multinomial Naive Bayes
   → alpha=0.1 (Laplace smoothing)
   ↓
Prediction: SPAM / HAM + Probability
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, NumPy, Pandas |
| Backend | Python 3.11, Flask 3.0 |
| Frontend | HTML5, CSS3, Vanilla JS |
| Server | Gunicorn (WSGI) |
| Deployment | Heroku |

---

## 🚀 Local Development

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python model/train_model.py
```

### 5. Run the Flask server
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## ☁️ Heroku Deployment

### Step 1 — Login to Heroku
```bash
heroku login
```

### Step 2 — Create Heroku app
```bash
heroku create spamshield-yourname
```

### Step 3 — Initialize git & push
```bash
git init
git add .
git commit -m "Initial commit: SpamShield spam classifier"
git push heroku main
```

### Step 4 — Open your app
```bash
heroku open
```

### Step 5 — View logs
```bash
heroku logs --tail
```

---

## 📡 API Reference

### `POST /predict`
Classify a single message.

**Request:**
```json
{
  "message": "WINNER!! You have won a £900 prize!",
  "type": "email"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "is_spam": true,
  "spam_probability": 97.3,
  "ham_probability": 2.7,
  "confidence": "Very High",
  "word_count": 8,
  "char_count": 38,
  "indicators": ["ALL CAPS usage", "Money/prize keywords", "Exclamation marks"],
  "message_type": "email",
  "timestamp": "2024-01-15 10:30:00"
}
```

### `POST /batch`
Classify up to 50 messages at once.

**Request:**
```json
{
  "messages": ["Message 1", "Message 2", "..."],
  "type": "sms"
}
```

### `GET /stats`
Get model performance metrics.

### `GET /health`
Health check endpoint.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~97% |
| Recall | ~97% |
| F1 Score | ~97% |

---

## 👤 Author

**Ravinder Singh**  
B.Tech CSE | Machine Learning & Python Developer  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

MIT License — free to use for academic and commercial projects.
