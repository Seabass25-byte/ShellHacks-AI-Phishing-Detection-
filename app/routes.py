import re
import torch
from flask import Blueprint, request, jsonify, render_template
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pickle
from lime.lime_text import LimeTextExplainer
from transformers import pipeline

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

PHISHING_THRESHOLD = 0.8

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

@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    content = request.json.get('content', '')
    urls = extract_urls(content)
    
    total_phishing_confidence = 0.0
    total_legitimate_confidence = 0.0
    count = 0
    result_data = {}

    # Process URLs
    if urls:
        for url in urls:
            phishing_conf, legitimate_conf = classify_url(url)
            total_phishing_confidence += phishing_conf * 100
            total_legitimate_confidence += legitimate_conf * 100
            count += 1

    # Process email content
    content_without_urls = re.sub(r'(http[s]?://\S+|www\.\S+)', '', content).strip()
    if content_without_urls:
        email_phishing_conf, email_legitimate_conf = classify_email(content_without_urls)
        total_phishing_confidence += email_phishing_conf * 100
        total_legitimate_confidence += email_legitimate_conf * 100
        count += 1

        # Generate simplified explanation
        lime_features = generate_lime_explanation(content_without_urls, email_classifier, email_vectorizer, "phishing" if total_phishing_confidence > total_legitimate_confidence else "legitimate")

    # Calculate final confidence scores
    if count > 0:
        combined_phishing_confidence = total_phishing_confidence / count
        combined_legitimate_confidence = total_legitimate_confidence / count
    else:
        combined_phishing_confidence = combined_legitimate_confidence = 0

    # Determine final classification
    final_classification = "phishing" if combined_phishing_confidence > PHISHING_THRESHOLD * 100 else "legitimate"

    # Generate educational tips
    educational_tips = generate_educational_tips(final_classification)

    # Prepare response
    result_data['final_result'] = final_classification
    result_data['phishing_confidence'] = combined_phishing_confidence
    result_data['legitimate_confidence'] = combined_legitimate_confidence
    result_data['explanation'] = lime_features
    result_data['tips'] = educational_tips

    return jsonify(result_data)
