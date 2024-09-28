import pickle

with open('./model/email_vectorization.pk1', 'rb') as f:
    vectorizer = pickle.load(f)

with open('./model/email_classifier.pk1', 'rb') as f:
    classifier = pickle.load(f)  


def classify_email(email_body):
    email_vector = vectorizer.transform([email_body])
    prediction = classifier.predict([email_vector])
    return prediction[0] # Either legitimate or spam
    
