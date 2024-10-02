import re
import torch
from flask import Blueprint, request, jsonify, render_template
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pickle
from lime.lime_text import LimeTextExplainer
import requests
import os
from dotenv import load_dotenv

# Set up Flask blueprint
api_bp = Blueprint('api_bp', __name__)

# Load models and vectorizers
url_model = DistilBertForSequenceClassification.from_pretrained('./results/checkpoint-1000')
url_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url_model.to(device)

with open('./model/email_classifier.pkl', 'rb') as f:
    email_classifier = pickle.load(f)

with open('./model/email_vectorizer.pkl', 'rb') as f:
    email_vectorizer = pickle.load(f)

# Load VirusTotal API key from environment
load_dotenv()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

if not VIRUSTOTAL_API_KEY:
    raise ValueError("No VirusTotal API key found in .env")

PHISHING_THRESHOLD = 0.8

# VirusTotal API check
def check_url_with_virustotal(url_to_check):
    """Check a URL with VirusTotal."""
    url = "https://www.virustotal.com/vtapi/v2/url/report"
    params = {
        'apikey': VIRUSTOTAL_API_KEY,
        'resource': url_to_check
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response_code'] == 1:  # URL found in VirusTotal's database
            if data['positives'] > 0:
                return {
                    'status': 'malicious',
                    'positives': data['positives'],
                    'total': data['total'],
                    'message': f"{data['positives']}/{data['total']} vendors flagged this URL as malicious."
                }
            else:
                return {
                    'status': 'clean',
                    'message': "No vendors flagged this URL as malicious."
                }
        else:
            return {
                'status': 'not_found',
                'message': "URL not found in VirusTotal database."
            }
    else:
        return {
            'status': 'error',
            'message': "Error checking URL with VirusTotal."
        }

# Helper functions for LIME and classification
def extract_urls(content):
    url_pattern = re.compile(r'(http[s]?://\S+|www\.\S+)')
    return url_pattern.findall(content)

def generate_lime_explanation(content, model, vectorizer, classification):
    explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
    def predict_proba(texts):
        text_vectorized = vectorizer.transform(texts)
        return model.predict_proba(text_vectorized)
    
    explanation = explainer.explain_instance(content, predict_proba, num_features=5)
    
    simplified_explanation = f"This email has been classified as {classification}. Here's why:\n"
    for feature, importance in explanation.as_list():
        if importance > 0:
            simplified_explanation += f"- The word '{feature}' suggests this email might be phishing.\n"
        else:
            simplified_explanation += f"- The absence of '{feature}' suggests this email might be legitimate.\n"
    
    return simplified_explanation

def classify_url(url):
    inputs = url_tokenizer(url, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = url_model(**inputs)
    confidence = torch.softmax(outputs.logits, dim=-1)
    return confidence[0][1].item(), confidence[0][0].item()

def classify_email(content):
    email_vectorized = email_vectorizer.transform([content])
    predicted_proba = email_classifier.predict_proba(email_vectorized)
    return predicted_proba[0][1], predicted_proba[0][0]

def generate_explanation(content, features, classification):
    sorted_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
    explanation = f"This email has been classified as {classification}. Here's why:\n\n"
    
    for feature, importance in sorted_features:
        if importance > 0:
            explanation += f"- The presence of '{feature}' suggests it might be {classification} (importance: {importance:.2f})\n"
        else:
            explanation += f"- The absence of '{feature}' suggests it might be legitimate (importance: {abs(importance):.2f})\n"
    
    explanation += "\nAdditional analysis:\n"
    
    if classification == "phishing":
        explanation += "- Be cautious of emails asking for immediate action or verification.\n"
        explanation += "- Check the sender's email address carefully.\n"
        explanation += "- Hover over links (don't click) to see where they actually lead.\n"
        explanation += "- Be wary of generic greetings like 'Dear Customer'.\n"
    else:
        explanation += "- This email appears to be legitimate, but always stay vigilant.\n"
        explanation += "- If you're unsure, contact the sender through a known, trusted method.\n"
    
    return explanation

def generate_educational_tips(classification):
    if classification == "phishing":
        return [
            "Be cautious of emails asking for immediate action or verification.",
            "Check the sender's email address carefully.",
            "Hover over links (don't click) to see where they actually lead.",
            "Be wary of generic greetings like 'Dear Customer'."
        ]
    else:
        return [
            "This email appears legitimate, but always stay vigilant.",
            "If you're unsure, contact the sender through a known, trusted method."
        ]

# Main route for the web application
@api_bp.route('/')
def index():
    return render_template('index.html')

# API endpoint for predictions
@api_bp.route('/predict', methods=['POST'])
def predict():
    content = request.json.get('content', '')
    urls = extract_urls(content)
    
    total_phishing_confidence = 0.0
    total_legitimate_confidence = 0.0
    count = 0
    result_data = {}
    lime_features = []  # Initialize lime_features to avoid UnboundLocalError

    # Process URLs and check with VirusTotal
    if urls:
        for url in urls:
            phishing_conf, legitimate_conf = classify_url(url)
            total_phishing_confidence += phishing_conf * 100
            total_legitimate_confidence += legitimate_conf * 100
            count += 1

            # Check with VirusTotal
            vt_result = check_url_with_virustotal(url)
            result_data['virustotal'] = vt_result['message']

    # Process email content
    content_without_urls = re.sub(r'(http[s]?://\S+|www\.\S+)', '', content).strip()
    
    if content_without_urls:
        email_phishing_conf, email_legitimate_conf = classify_email(content_without_urls)
        total_phishing_confidence += email_phishing_conf * 100
        total_legitimate_confidence += email_legitimate_conf * 100
        count += 1

        # Generate LIME explanation
        lime_features = generate_lime_explanation(content_without_urls, email_classifier, email_vectorizer, "phishing" if total_phishing_confidence > PHISHING_THRESHOLD * 100 else "legitimate")

    # Calculate final confidence scores
    if count > 0:
        combined_phishing_confidence = total_phishing_confidence / count
        combined_legitimate_confidence = total_legitimate_confidence / count
    else:
        combined_phishing_confidence = combined_legitimate_confidence = 0

    # Determine final classification
    final_classification = "phishing" if combined_phishing_confidence > PHISHING_THRESHOLD * 100 else "legitimate"

    # Generate explanation
    explanation = lime_features if lime_features else "No explanation available due to insufficient content."
    result_data['explanation'] = explanation
    result_data['final_result'] = final_classification
    result_data['phishing_confidence'] = combined_phishing_confidence
    result_data['legitimate_confidence'] = combined_legitimate_confidence

    return jsonify(result_data)
