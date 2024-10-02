# PhishGuard - AI-Powered Phishing Detection Tool

![PhishGuard Logo](link-to-logo-image-if-available)

**PhishGuard** is an AI-powered phishing detection tool designed to help users identify phishing emails in real time. What makes PhishGuard unique is its ability not only to flag phishing attempts but also to educate users on *why* an email is marked as phishing, empowering them to recognize future attacks.

## 🛠 Features

- **Real-Time Phishing Detection**: Analyzes email content and URLs to identify phishing attempts using AI models.
- **Educational Feedback**: Provides users with detailed explanations on why an email is flagged as phishing, helping users learn how to spot phishing on their own.
- **URL Verification with VirusTotal**: Integrates the VirusTotal API to check URLs within emails for malicious content, enhancing detection accuracy.
- **User-Friendly Interface**: Simple and intuitive interface that makes it easy for anyone to use, regardless of technical background.

## 🚀 Technologies Used

- **Backend**: Flask
- **Model Training**: Scikit-learn, DistilBERT for phishing detection model
- **URL Analysis**: VirusTotal API
- **Frontend**: HTML, CSS (optional to include more if frontend tech was used)
- **Other Libraries**: 
  - LIME for model explainability
  - Requests for API handling
  - Torch for handling AI models (DistilBERT)

## 🎯 How It Works

PhishGuard works by analyzing both the **email content** and any **URLs** within the email. The system uses a combination of AI-powered models and third-party verification (VirusTotal) to flag suspicious content. Once a potential phishing attempt is detected, PhishGuard provides real-time feedback, educating the user on why the email was flagged.

### Key Components:
1. **Phishing Detection**: 
   - **Email Content**: Uses an AI model trained on phishing emails to detect suspicious language and patterns.
   - **URL Analysis**: Verifies URLs using the VirusTotal API to identify known malicious links.
   
2. **Educational Insights**:
   - PhishGuard doesn’t just flag emails—it provides detailed explanations, showing users the specific features that triggered the phishing flag. This helps users learn and become more aware of phishing tactics.

## 📝 Project Structure

```bash
├── app
│   ├── static/          # Static files (CSS, images, etc.)
│   ├── templates/       # HTML templates
│   ├── __init__.py      # Initialize Flask app
│   ├── routes.py        # Define routes
│   ├── run.py           # Main file to run the application
│
├── model/               # Model-related files
│   ├── email_classifier.pkl   # Trained phishing detection model
│   ├── email_vectorizer.pkl   # Vectorizer for email content
│
├── data/                # Data folder (sample emails, datasets)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .env                 # Environment variables (API keys, config)
