from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Sample data with email and URL images
images = [
    {"url": "https://www.imperva.com/learn/wp-content/uploads/sites/13/2019/01/phishing-attack-email-example.png", "is_phishing": True},   # phishing email
    {"url": "https://security.virginia.edu/sites/security/files/styles/scale_600/public/phish%20image_0.png?itok=rVkQtFHV", "is_phishing": False},    # legitimate URL
    {"url": "https://phishing.iu.edu/images/edts_Phishing%20Example%20amazon.png", "is_phishing": False},  # legitimate email
    {"url": "https://blog.usecure.io/hubfs/Example%20of%20an%20email%20account%20upgrade%20scam.jpg", "is_phishing": True},     # phishing URL
]

@app.route('/')
def game():
    return render_template('game.html')

@app.route('/get_all_images', methods=['GET'])
def get_all_images():
    return jsonify(images)

@app.route('/submit_guess', methods=['POST'])
def submit_guess():
    data = request.get_json()
    user_guess = data['guess']
    actual = data['is_phishing']
    
    result = "correct" if user_guess == actual else "incorrect"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
