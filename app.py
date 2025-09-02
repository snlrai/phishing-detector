import joblib
import pandas as pd
import numpy as np
import re
import string
from flask import Flask, request, jsonify
from scipy.sparse import hstack

#Initialize Flask App
app = Flask(__name__)


try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model, vectorizer = None, None


suspicious_keywords = [
    "verify", "password", "urgent", "account", "login", "bank", 
    "limited", "security", "update", "confirm"
]

def engineer_features(email_text):
    if not isinstance(email_text, str):
        email_text = ""
    features = {}
    features['word_count'] = len(email_text.split())
    features['char_count'] = len(email_text)
    features['keyword_count'] = sum(1 for keyword in suspicious_keywords if keyword in email_text.lower())
    features['link_count'] = len(re.findall(r'http[s]?://', email_text))
    uppercase_chars = sum(1 for char in email_text if char.isupper())
    features['uppercase_ratio'] = uppercase_chars / features['char_count'] if features['char_count'] > 0 else 0
    features['punctuation_count'] = sum(1 for char in email_text if char in string.punctuation)
    features['number_count'] = sum(1 for char in email_text if char.isdigit())
    return features

#Define Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    data = request.get_json()
    if not data or 'email_text' not in data:
        return jsonify({'error': 'Missing email_text field'}), 400
    
    email_text = data['email_text']

    #Create two sets of features
    engineered_features_dict = engineer_features(email_text)
    engineered_features_df = pd.DataFrame([engineered_features_dict])
    
    text_vector = vectorizer.transform([email_text])
    
    # Combine features
    final_features = hstack([text_vector, engineered_features_df.values])
    
    # Make prediction
    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)

    # Return result
    result = {
        'is_phishing': bool(prediction[0]),
        'phishing_probability': f"{prediction_proba[0][1]:.4f}" # Probability of it being phishing
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)