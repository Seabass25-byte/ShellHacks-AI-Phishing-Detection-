import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./model/phishing_model')
tokenizer = DistilBertTokenizer.from_pretrained('./model/phishing_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_url(url):
    inputs = tokenizer(url, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()  # 0 = benign, 1 = phishing
