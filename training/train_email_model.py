import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from email_utils import load_emails



# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
legit_dir = os.path.join(data_dir, "legitimate")
phishing_dir = os.path.join(data_dir, "phishing")
model_dir = os.path.join(script_dir, "..", "model")

# Print debugging information
print(f"Script directory: {script_dir}")
print(f"Data directory: {data_dir}")
print(f"Legitimate email directory: {legit_dir}")
print(f"Phishing email directory: {phishing_dir}")

# Check if directories exist
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
if not os.path.exists(legit_dir):
    raise FileNotFoundError(f"Legitimate email directory does not exist: {legit_dir}")
if not os.path.exists(phishing_dir):
    raise FileNotFoundError(f"Phishing email directory does not exist: {phishing_dir}")

# Create the model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

try:
    print("Loading legitimate emails...")
    legit_emails = load_emails(legit_dir, label="legitimate")
    print(f"Loaded {len(legit_emails)} legitimate emails.")

    print("Loading phishing emails...")
    phishing_emails = load_emails(phishing_dir, label="phishing")
    print(f"Loaded {len(phishing_emails)} phishing emails.")

    # Balance the dataset to avoid bias
    legit_emails = legit_emails[:len(phishing_emails)]

    # Combine all emails and their labels
    all_emails = legit_emails + phishing_emails

    # Extract the email texts (ignore the labels) for vectorization
    email_texts = [email_text for email_text, label in all_emails]

    # Extract the labels separately
    labels = [label for email_text, label in all_emails]

    # Split the data into training and testing sets for better evaluation
    X_train, X_test, y_train, y_test = train_test_split(email_texts, labels, test_size=0.2, random_state=42)

    # Vectorize the email texts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the Naive Bayes classifier with alpha smoothing
    classifier = MultinomialNB(alpha=0.5)
    classifier.fit(X_train_vec, y_train)

    # Predictions on the test set
    y_pred = classifier.predict(X_test_vec)

    # Calculate accuracy and show the confusion matrix and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


    # Save the vectorizer and the trained model to the 'model' directory
    vectorizer_path = os.path.join(model_dir, 'email_vectorizer.pkl')
    classifier_path = os.path.join(model_dir, 'email_classifier.pkl')

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)

    print(f"Model and vectorizer saved in 'model' directory as 'email_classifier.pkl' and 'email_vectorizer.pkl'.")


except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise

