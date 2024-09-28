import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from flask import Blueprint, request, jsonify

# Create a Blueprint for routes
api_bp = Blueprint('api_bp', __name__)

# Load the Naive Bayes email classifier and vectorizer
with open('./model/email_classifier.pkl', 'rb') as f:
    email_classifier = pickle.load(f)

with open('./model/email_vectorizer.pkl', 'rb') as f:
    email_vectorizer = pickle.load(f)

# Load the pre-trained DistilBERT model and tokenizer for URLs using safetensors
url_model = DistilBertForSequenceClassification.from_pretrained('./results/checkpoint-1000', use_safetensors=True)
url_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Ensure the model is on the correct device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url_model.to(device)

# Route for email phishing detection
@api_bp.route('/predict-email', methods=['POST'])
def predict_email():
    email_content = request.json.get('email')

    # Vectorize the email content
    email_vectorized = email_vectorizer.transform([email_content])

    # Make a prediction using the Naive Bayes classifier
    prediction = email_classifier.predict(email_vectorized)
    result = "phishing" if prediction[0] == 'phishing' else "legitimate"

    return jsonify({'result': result})

# Route for URL phishing detection
@api_bp.route('/predict-url', methods=['POST'])
def predict_url():
    url = request.json.get('url')

    # Tokenize the URL using DistilBERT tokenizer
    inputs = url_tokenizer(url, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = url_model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    result = "phishing" if prediction == 1 else "legitimate"

    return jsonify({'result': result})
