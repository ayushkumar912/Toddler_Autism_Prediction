from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib 

app = Flask(__name__)
CORS(app)

print("Loading model and label encoders...")
model = joblib.load('naive_bayes_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
print("Model and label encoders loaded successfully.\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    print("Incoming data:", data)

   
    required_keys = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD',
                     'Who completed the test', 'A1_Score', 'A2_Score', 
                     'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 
                     'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 
                     'Age_Mons']
    
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'Missing key: {key}', 'message': 'One or more input values are invalid.'}), 400

    try:
       
        sex_encoded = label_encoders['Sex'].transform([data['Sex']])[0]
        ethnicity_encoded = label_encoders['Ethnicity'].transform([data['Ethnicity']])[0]
        jaundice_encoded = label_encoders['Jaundice'].transform([data['Jaundice']])[0]
        family_mem_with_ASD_encoded = label_encoders['Family_mem_with_ASD'].transform([data['Family_mem_with_ASD']])[0]
        who_completed_test_encoded = label_encoders['Who completed the test'].transform([data['Who completed the test']])[0]
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}', 'message': 'One or more input values are invalid.'}), 400
    except ValueError as e:
        return jsonify({'error': str(e), 'message': 'One or more input values are invalid.'}), 400

   
    try:
        qchat_score = sum(int(data[f'A{i}_Score']) for i in range(1, 11))
    except ValueError:
        return jsonify({'error': 'Score values must be integers.'}), 400

    
    input_data = {
        'A1': int(data['A1_Score']),
        'A2': int(data['A2_Score']),
        'A3': int(data['A3_Score']),
        'A4': int(data['A4_Score']),
        'A5': int(data['A5_Score']),
        'A6': int(data['A6_Score']),
        'A7': int(data['A7_Score']),
        'A8': int(data['A8_Score']),
        'A9': int(data['A9_Score']),
        'A10': int(data['A10_Score']),
        'Age_Mons': int(data['Age_Mons']),
        'Qchat-10-Score': qchat_score,
        'Sex': sex_encoded,
        'Ethnicity': ethnicity_encoded,
        'Jaundice': jaundice_encoded,
        'Family_mem_with_ASD': family_mem_with_ASD_encoded,
        'Who completed the test': who_completed_test_encoded
    }

    
    input_df = pd.DataFrame([input_data])

    
    try:
        prediction_proba = model.predict_proba(input_df)[:, 1]
        predicted_class = model.predict(input_df)[0]
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

    return jsonify({
        'prediction': int(predicted_class),
        'probability': float(prediction_proba)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
