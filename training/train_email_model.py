import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import load_emails

# Directories for legitimate and spam emails
legit_dir = os.path.join("..", "data", "legitimate")
spam_dir = os.path.join("..", "data", "spam")

# Load emails from directories
legit_emails = load_emails(legit_dir, label="legitimate")
spam_emails = load_emails(spam_dir, label="spam")

# Balance dataset by truncating legitimate emails to match spam count
legit_emails = legit_emails[:len(spam_emails)]

# Combine and label the emails
all_emails = legit_emails + spam_emails
labels = ["legitimate"] * len(legit_emails) + ["spam"] * len(spam_emails)

# Vectorize the emails using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_emails)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Save the vectorizer and the trained model
with open('../model/email_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('../model/email_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Model and vectorizer saved in 'model' directory.")