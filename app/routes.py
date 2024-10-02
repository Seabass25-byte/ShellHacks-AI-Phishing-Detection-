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
VIRUSTOTAL_MALICIOUS_WEIGHT = 70  # Weight when URL is flagged as malicious
VIRUSTOTAL_THRESHOLD = 3  # Threshold number of vendors to increase phishing confidence

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
            positives = data['positives']
            total = data['total']
            if positives > 0:
                return {
                    'status': 'malicious',
                    'positives': positives,
                    'total': total,
                    'weight': VIRUSTOTAL_MALICIOUS_WEIGHT if positives >= VIRUSTOTAL_THRESHOLD else 1,
                    'message': f"{positives}/{total} vendors flagged this URL as malicious."
                }
            else:
                return {
                    'status': 'clean',
                    'weight': 1,
                    'message': "No vendors flagged this URL as malicious."
                }
        else:
            return {
                'status': 'not_found',
                'weight': 1,
                'message': "URL not found in VirusTotal database."
            }
    else:
        return {
            'status': 'error',
            'weight': 1,
            'message': "Error checking URL with VirusTotal."
        }

# Helper functions for LIME and classification
def extract_urls(content):
    url_pattern = re.compile(r'(http[s]?://\S+|www\.\S+)')
    return url_pattern.findall(content)

def generate_improved_explanation(content, model, vectorizer, classification, threshold=0.8):
    explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
    def predict_proba(texts):
        text_vectorized = vectorizer.transform(texts)
        return model.predict_proba(text_vectorized)
    
    # Get LIME explanation
    exp = explainer.explain_instance(content, predict_proba, num_features=10)
    
    # Sort features by absolute importance
    features = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
    
    # Generate a more nuanced explanation
    confidence = exp.predict_proba[1] if classification == 'phishing' else exp.predict_proba[0]
    
    explanation = (f"This email has been classified as {classification} "
                  f"with {confidence:.1%} confidence.\n\n"
                  f"Key factors in this classification:\n")
    
    # Add positive and negative features separately
    phishing_indicators = [f for f in features if f[1] > 0]
    legitimate_indicators = [f for f in features if f[1] < 0]
    
    if phishing_indicators:
        explanation += "\nPotential phishing indicators:\n"
        for feature, importance in phishing_indicators[:3]:
            if feature.lower() in content.lower():
                explanation += f"- The presence of '{feature}' (importance: {importance:.2f})\n"
    
    if legitimate_indicators:
        explanation += "\nFactors suggesting legitimacy:\n"
        for feature, importance in legitimate_indicators[:3]:
            abs_importance = abs(importance)
            if feature.lower() in content.lower():
                explanation += f"- The presence of '{feature}' (importance: {abs_importance:.2f})\n"
    
    # Add confidence-based advice
    explanation += "\nRecommendation:\n"
    if confidence > threshold:
        if classification == 'phishing':
            explanation += "- This email shows strong indicators of being a phishing attempt.\n"
            explanation += "- Exercise extreme caution and do not click any links or download attachments.\n"
        else:
            explanation += "- This email appears to be legitimate with high confidence.\n"
            explanation += "- However, always remain vigilant and verify if something seems unusual.\n"
    else:
        explanation += "- The classification is not highly confident.\n"
        explanation += "- Treat this email with caution and verify through other channels if necessary.\n"
    
    return explanation

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

# Main route for the web application
@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json.get('content', '')
        if not content:
            return jsonify({'error': 'No content provided'}), 400

        urls = extract_urls(content)
        result_data = {}
        total_phishing_confidence = 0.0
        total_legitimate_confidence = 0.0
        count = 0

        malicious_url_detected = False

        # Process URLs and check VirusTotal
        if urls:
            for url in urls:
                phishing_conf, legitimate_conf = classify_url(url)
                total_phishing_confidence += phishing_conf
                total_legitimate_confidence += legitimate_conf
                count += 1

                # VirusTotal check
                vt_result = check_url_with_virustotal(url)
                result_data['virustotal'] = vt_result['message']

                if vt_result['status'] == 'malicious' and vt_result['positives'] > 10:
                    malicious_url_detected = True
                
                # Apply weight if malicious URL detected
                if vt_result['status'] == 'malicious':
                    total_phishing_confidence += phishing_conf * vt_result['weight']
                    total_legitimate_confidence -= legitimate_conf * vt_result['weight']

        # Process email content to provide analysis even if phishing detected
        content_without_urls = re.sub(r'(http[s]?://\S+|www\.\S+)', '', content).strip()
        if content_without_urls:
            email_phishing_conf, email_legitimate_conf = classify_email(content_without_urls)
            total_phishing_confidence += email_phishing_conf * 100
            total_legitimate_confidence += email_legitimate_conf * 100
            count += 1

            # Generate explanation even if phishing is detected
            final_classification = "phishing" if malicious_url_detected or total_phishing_confidence/count > PHISHING_THRESHOLD * 100 else "legitimate"
            explanation = generate_improved_explanation(
                content_without_urls,
                email_classifier,
                email_vectorizer,
                final_classification,
                PHISHING_THRESHOLD
            )
            result_data['explanation'] = explanation

        if malicious_url_detected:
            result_data.update({
                'final_result': "phishing",
                'phishing_confidence': 100.0,
                'legitimate_confidence': 0.0,
                'explanation': explanation,
                'recommendation': "The URL in this email was flagged by multiple vendors as malicious. This email has been classified as phishing. Do not click on any links or provide any personal information."
            })
        elif count > 0:
            result_data.update({
                'final_result': "phishing" if total_phishing_confidence/count > PHISHING_THRESHOLD * 100 else "legitimate",
                'phishing_confidence': total_phishing_confidence/count,
                'legitimate_confidence': total_legitimate_confidence/count,
                'explanation': explanation,
                'recommendation': "The classification is not highly confident. Treat this email with caution and verify through other channels if necessary."
            })
        else:
            return jsonify({'error': 'Could not process content'}), 400

        return jsonify(result_data)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
