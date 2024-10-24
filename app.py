from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model_nb = joblib.load('gaussian_nb_model.joblib')
model_rf = joblib.load('random_forest_model.joblib')
ensemble_model = joblib.load('ensemble_model.joblib')
label_encoders = joblib.load('optimized_label_encoders.joblib')
scaler = joblib.load('scaler.joblib')  

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    df = pd.DataFrame([input_data])

    for col in ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    scaled_features = scaler.transform(df[['A1', 'A2', 'A3', 'A4', 'A5', 
                                            'A6', 'A7', 'A8', 'A9', 
                                            'A10', 'Age_Mons', 
                                            'Qchat-10-Score', 
                                            'Sex', 'Ethnicity', 
                                            'Jaundice', 'Family_mem_with_ASD']])
    
    prediction_nb = model_nb.predict(scaled_features)
    prediction_rf = model_rf.predict(scaled_features)
    prediction_ensemble = ensemble_model.predict(scaled_features)
    
    return jsonify({
        'Naive_Bayes_Prediction': prediction_nb.tolist(),
        'Random_Forest_Prediction': prediction_rf.tolist(),
        'Ensemble_Prediction': prediction_ensemble.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
