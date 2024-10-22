from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(filename='app.log', level=logging.INFO)

try:
    print("Loading models and transformers...")
    voting_classifier = joblib.load('optimized_voting_classifier.joblib')
    label_encoders = joblib.load('optimized_label_encoders.joblib')
    scaler = joblib.load('optimized_scaler.joblib')
    print("Models and transformers loaded successfully.\n")
except Exception as e:
    print(f"Error loading models/transformers: {str(e)}")
    voting_classifier, label_encoders, scaler = None, None, None

print('hi')

@app.route('/')
def home():
    return render_template('index.html')
print("")

@app.route('/predict', methods=['POST'])
def predict():
    if voting_classifier is None or label_encoders is None or scaler is None:
        return jsonify({'error': 'Models or transformers not loaded properly'}), 500

    data = request.get_json()
    logging.info(f"Incoming data: {data}")

    required_keys = [
        'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD',
        'Who completed the test', 'A1_Score', 'A2_Score', 
        'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 
        'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 
        'Age_Mons'
    ]
    
    # Validate required keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return jsonify({
            'error': f'Missing keys: {", ".join(missing_keys)}',
            'message': 'One or more input values are missing.'
        }), 400

    try:
        # Encode categorical variables
        encoded_data = {
            'Sex': label_encoders['Sex'].transform([data['Sex']])[0],
            'Ethnicity': label_encoders['Ethnicity'].transform([data['Ethnicity']])[0],
            'Jaundice': label_encoders['Jaundice'].transform([data['Jaundice']])[0],
            'Family_mem_with_ASD': label_encoders['Family_mem_with_ASD'].transform([data['Family_mem_with_ASD']])[0],
            'Who completed the test': label_encoders['Who completed the test'].transform([data['Who completed the test']])[0]
        }
    except (KeyError, ValueError) as e:
        return jsonify({
            'error': 'Encoding error',
            'message': f'Error encoding categorical variables: {str(e)}'
        }), 400

    try:
        # Process behavioral scores
        behavior_scores = {}
        for i in range(1, 11):
            score = int(data[f'A{i}_Score'])
            if not (0 <= score <= 1):  # Assuming binary scores
                return jsonify({
                    'error': f'Invalid score for A{i}',
                    'message': 'Behavioral scores must be 0 or 1'
                }), 400
            behavior_scores[f'A{i}'] = score

        # Calculate behavioral score
        behavioral_score = sum(behavior_scores.values())
        
        # Process continuous features
        continuous_features = {
            'Age_Mons': float(data['Age_Mons']),
            'Behavioral_Score': behavioral_score,
            'Qchat-10-Score': behavioral_score  # In this case, same as behavioral score
        }

        # Scale continuous features
        scaled_features = scaler.transform([[
            continuous_features['Age_Mons'],
            continuous_features['Qchat-10-Score'],
            continuous_features['Behavioral_Score']
        ]])[0]

        # Combine all features
        input_data = {
            'Age_Mons': scaled_features[0],
            'Qchat-10-Score': scaled_features[1],
            'Behavioral_Score': scaled_features[2],
            **behavior_scores,
            **encoded_data
        }
        print("Hi")

        input_df = pd.DataFrame([input_data])

    except ValueError as e:
        return jsonify({
            'error': 'Data processing error',
            'message': f'Error processing input data: {str(e)}'
        }), 400

    try:
        # Get predictions from both models through voting classifier
        prediction_proba = voting_classifier.predict_proba(input_df)
        predicted_class = voting_classifier.predict(input_df)[0]

        # Get individual model probabilities
        individual_probas = [
            estimator.predict_proba(input_df)[:, 1][0] 
            for estimator in voting_classifier.estimators_
        ]

        result = {
            'prediction': 'May have' if int(predicted_class) == 1 else 'No',
            'probability': f"{float(prediction_proba[:, 1][0]) * 100:.3f}%",
            'gaussian_probability': f"{float(individual_probas[0]) * 100:.3f}%",
            'bernoulli_probability': f"{float(individual_probas[1]) * 100:.3f}%"
        }

        logging.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500
logging.info("Received data for prediction.")
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5040, debug=True)