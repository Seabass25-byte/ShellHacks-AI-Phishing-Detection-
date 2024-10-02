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
    for idx, filename in enumerate(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                # Attempt to open the file using utf-8 first, then fallback to latin-1
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                except UnicodeDecodeError:
                    with open(filepath, 'r', encoding='latin-1') as file:
                        content = file.read()

                # Split headers and body
                parts = content.split("\n\n", 1)
                body = parts[1] if len(parts) > 1 else content

                # Skip if body is empty
                if not body.strip():
                    print(f"Empty email body in {filename}")
                    continue

                # Preprocess and add to emails list
                emails.append((preprocess_text(body), label))

                # Print progress every 1000 emails
                if idx % 1000 == 0:
                    print(f"Processed {idx} emails.")

            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return emails





