import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from email_utils import load_emails

# Directories for legitimate and phishing emails
legit_dir = os.path.join("..", "data", "legitimate")
phishing_dir = os.path.join("..", "data", "phishing")

# Create model directory if it doesn't exist
model_dir = os.path.join("..", "model")
os.makedirs(model_dir, exist_ok=True)

print("Loading legitimate emails...")
legit_emails = load_emails(legit_dir, label="legitimate")
print(f"Loaded {len(legit_emails)} legitimate emails.")

print("Loading phishing emails...")
phishing_emails = load_emails(phishing_dir, label="phishing")
print(f"Loaded {len(phishing_emails)} phishing emails.")

# To balance the dataset, only use as many legitimate emails as there are phishing emails
legit_emails = legit_emails[:len(phishing_emails)]

# Combine all emails and their labels
all_emails = legit_emails + phishing_emails

# Extract the email texts (ignore the labels) for vectorization
email_texts = [email_text for email_text, label in all_emails]

# Extract the labels separately
labels = [label for email_text, label in all_emails]

# Vectorize the email texts
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(email_texts)

# Train the Naive Bayes classifier with alpha smoothing
classifier = MultinomialNB(alpha=0.5)
classifier.fit(X, labels)

# Predictions
predictions = classifier.predict(X)

# Calculate accuracy and show the confusion matrix and classification report
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_matrix(labels, predictions))
print("Classification Report:\n", classification_report(labels, predictions))

# Save the vectorizer and the trained model to the 'model' directory
vectorizer_path = os.path.join(model_dir, 'email_vectorizer.pkl')
classifier_path = os.path.join(model_dir, 'email_classifier.pkl')

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

with open(classifier_path, 'wb') as f:
    pickle.dump(classifier, f)

print(f"Model and vectorizer saved in 'model' directory as 'email_classifier.pkl' and 'email_vectorizer.pkl'.")
