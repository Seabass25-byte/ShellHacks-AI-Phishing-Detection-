import re
import torch
from flask import Blueprint, request, jsonify
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

gpt_model = pipeline("text-generation", model="distilgpt2")

PHISHING_THRESHOLD = 0.8

def extract_urls(content):
    url_pattern = re.compile(r'(http[s]?://\S+|www\.\S+)')
    return url_pattern.findall(content)

def generate_lime_explanation(content, model, vectorizer):
    explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
    def predict_proba(texts):
        text_vectorized = vectorizer.transform(texts)
        return model.predict_proba(text_vectorized)
    
    explanation = explainer.explain_instance(content, predict_proba, num_features=5)
    return explanation.as_list()

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
    # Sort features by absolute importance
    sorted_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
    
    explanation = f"This email has been classified as {classification}. Here's why:\n\n"
    
    for feature, importance in sorted_features:
        if importance > 0:
            explanation += f"- The presence of '{feature}' suggests it might be {classification} (importance: {importance:.2f})\n"
        else:
            explanation += f"- The absence of '{feature}' suggests it might be {'legitimate' if classification == 'phishing' else 'phishing'} (importance: {abs(importance):.2f})\n"
    
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

        lime_features = generate_lime_explanation(content_without_urls, email_classifier, email_vectorizer)

    # Calculate final confidence scores
    if count > 0:
        combined_phishing_confidence = total_phishing_confidence / count
        combined_legitimate_confidence = total_legitimate_confidence / count
    else:
        combined_phishing_confidence = combined_legitimate_confidence = 0

    # Determine final classification
    final_classification = "phishing" if combined_phishing_confidence > PHISHING_THRESHOLD * 100 else "legitimate"

    # Generate explanation
    if content_without_urls:
        explanation = generate_explanation(content_without_urls, lime_features, final_classification)
        result_data['explanation'] = explanation

    result_data['final_result'] = final_classification
    result_data['phishing_confidence'] = combined_phishing_confidence
    result_data['legitimate_confidence'] = combined_legitimate_confidence

    return jsonify(result_data)