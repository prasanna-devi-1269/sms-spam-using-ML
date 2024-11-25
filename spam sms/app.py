# app.py

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = [request.form['message']]
    features = vectorizer.transform(message)
    prediction = model.predict(features)
    return render_template('index.html', prediction_text=f'Spam Prediction: {"Spam" if prediction[0] else "Ham"}')

if __name__ == '__main__':
    app.run(debug=True)
