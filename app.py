"""
Email & SMS Spam Classifier - Flask Backend
Author: Ravinder Singh
"""

import os
import re
import pickle
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'spam_classifier.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'metrics.pkl')

pipeline = None
metrics = {}

def load_model():
    global pipeline, metrics
    try:
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        with open(METRICS_PATH, 'rb') as f:
            metrics = pickle.load(f)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model not found, training now...")
        from model.train_model import train
        pipeline, metrics = train()

load_model()


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'\b\d{10,}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', 'phone', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_spam_indicators(text: str) -> list:
    """Return list of suspicious patterns found in the message."""
    indicators = []
    lower = text.lower()
    patterns = {
        "Contains URL": r'http\S+|www\S+|bit\.ly',
        "Contains phone number": r'\b\d{10,}\b|\b0800\b',
        "ALL CAPS usage": r'\b[A-Z]{4,}\b',
        "Money/prize keywords": r'\b(win|prize|cash|reward|claim|bonus|gift|voucher|lottery)\b',
        "Urgency keywords": r'\b(urgent|free|limited|hurry|now|today|expire|alert)\b',
        "Suspicious keywords": r'\b(congratulations|selected|awarded|offer|click|subscribe)\b',
        "Exclamation marks": r'!{1,}',
        "Currency symbols": r'[$£€¥]',
    }
    for name, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            indicators.append(name)
    return indicators


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data.get('message', '').strip()
    msg_type = data.get('type', 'email')  # 'email' or 'sms'

    if not message:
        return jsonify({'error': 'Message is required'}), 400
    if len(message) > 5000:
        return jsonify({'error': 'Message too long (max 5000 chars)'}), 400

    cleaned = preprocess(message)
    prediction = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]

    spam_prob = float(proba[list(pipeline.classes_).index('spam')])
    ham_prob = float(proba[list(pipeline.classes_).index('ham')])

    # Confidence label
    if spam_prob >= 0.85:
        confidence_label = "Very High"
    elif spam_prob >= 0.65:
        confidence_label = "High"
    elif spam_prob >= 0.45:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    indicators = get_spam_indicators(message) if prediction == 'spam' else []

    response = {
        'prediction': prediction,
        'is_spam': prediction == 'spam',
        'spam_probability': round(spam_prob * 100, 2),
        'ham_probability': round(ham_prob * 100, 2),
        'confidence': confidence_label,
        'message_type': msg_type,
        'word_count': len(message.split()),
        'char_count': len(message),
        'indicators': indicators,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    logger.info(f"[{msg_type.upper()}] Prediction={prediction} | Spam%={spam_prob:.2%}")
    return jsonify(response)


@app.route('/batch', methods=['POST'])
def batch_predict():
    data = request.get_json(force=True)
    messages = data.get('messages', [])
    msg_type = data.get('type', 'email')

    if not messages:
        return jsonify({'error': 'messages array is required'}), 400
    if len(messages) > 50:
        return jsonify({'error': 'Max 50 messages per batch'}), 400

    results = []
    for i, msg in enumerate(messages):
        text = str(msg).strip()
        if not text:
            results.append({'index': i, 'error': 'Empty message'})
            continue
        cleaned = preprocess(text)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        spam_prob = float(proba[list(pipeline.classes_).index('spam')])
        results.append({
            'index': i,
            'message_preview': text[:80] + ('...' if len(text) > 80 else ''),
            'prediction': pred,
            'is_spam': pred == 'spam',
            'spam_probability': round(spam_prob * 100, 2),
        })

    spam_count = sum(1 for r in results if r.get('is_spam'))
    return jsonify({
        'results': results,
        'summary': {
            'total': len(results),
            'spam': spam_count,
            'ham': len(results) - spam_count,
            'spam_rate': round(spam_count / len(results) * 100, 2) if results else 0,
        }
    })


@app.route('/stats')
def stats():
    return jsonify({
        'model': 'Multinomial Naive Bayes + TF-IDF',
        'metrics': metrics,
        'status': 'ready',
        'version': '1.0.0',
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': pipeline is not None})


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
