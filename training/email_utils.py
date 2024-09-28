import os
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

def preprocess_text(text):
    """
    Preprocess the text by performing the following:
    - Lowercasing
    - Removing special characters and punctuation
    - Removing numbers
    - Removing stopwords
    - Stemming words
    Args:
        text (str): The input email text.
    Returns:
        str: The preprocessed text.
    """
    # Lowercase the text
    text = text.lower()

    # Remove special characters, punctuation, and numbers
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

    # Stem the words
    tokens = [ps.stem(word) for word in tokens]

    # Rejoin the tokens
    return ' '.join(tokens)


def load_emails(directory, label):
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    content = file.read()
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1:
                        body = parts[1]
                    else:
                        body = content
                    emails.append((preprocess_text(body), label))  # Add label to each email
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return emails



