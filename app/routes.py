from flask import Blueprint, request, jsonify
import pickle
import torch
import re
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

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

# Threshold value for classifying phishing
PHISHING_THRESHOLD = 0.9  # You can tune this value

# Helper function to extract URLs from the input
def extract_urls(content):
    url_pattern = re.compile(r'(http[s]?://\S+|www\.\S+)')
    urls = url_pattern.findall(content)
    return urls

# Unified route for email and URL phishing detection
@api_bp.route('/predict', methods=['POST'])
def predict():
    content = request.json.get('content')

    # Extract URLs from the content
    urls = extract_urls(content)

    # Initialize result dictionary
    result_data = {}

    # If URLs are found, process them separately
    if urls:
        url_results = []
        for url in urls:
            # Tokenize and classify the URL
            inputs = url_tokenizer(url, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = url_model(**inputs)
            
            logits = outputs.logits
            confidence = torch.softmax(logits, dim=-1)
            phishing_confidence = confidence[0][1].item() * 100
            legitimate_confidence = confidence[0][0].item() * 100

            url_results.append({
                'url': url, 
                'phishing_confidence': phishing_confidence, 
                'legitimate_confidence': legitimate_confidence
            })

        result_data['url_results'] = url_results

    # Remove URLs from content for email processing
    content_without_urls = re.sub(r'(http[s]?://\S+|www\.\S+)', '', content).strip()

    # If there is still content left (after removing URLs), treat it as email content
    if content_without_urls:
        email_vectorized = email_vectorizer.transform([content_without_urls])
        predicted_proba = email_classifier.predict_proba(email_vectorized)

        # Extract probabilities for legitimate and phishing
        phishing_confidence = predicted_proba[0][1] * 100  # Probability of phishing
        legitimate_confidence = predicted_proba[0][0] * 100  # Probability of legitimate

        # Classify based on the phishing confidence threshold
        if phishing_confidence > PHISHING_THRESHOLD * 100:
            result_data['email_result'] = "phishing"
        else:
            result_data['email_result'] = "legitimate"

        result_data['phishing_confidence'] = phishing_confidence
        result_data['legitimate_confidence'] = legitimate_confidence

    return jsonify(result_data)
