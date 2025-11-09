import os
from flask import Flask, request, jsonify
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and preprocessor
model = xgb.XGBClassifier()
model.load_model('readmission_model.json')
prep = joblib.load('preprocessor_fixed.pkl')  # Use the fixed preprocessor

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        X = prep.transform(df)
        
        # Convert to dense array if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Predict
        risk = float(model.predict_proba(X)[0][1])
        
        return jsonify({
            "readmission_risk": round(risk, 3),
            "high_risk": risk > 0.5
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/features', methods=['GET'])
def features():
    """Return expected features for the model"""
    expected_features = {
        'numerical': ['age', 'length_of_stay', 'num_lab_procedures', 'num_medications', 'glucose_level'],
        'categorical': ['admission_type', 'primary_diagnosis']
    }
    return jsonify(expected_features)

if __name__ == '__main__': 
    port = int(os.environ.get('PORT', 5000))  # Railway provides PORT
    app.run(host='0.0.0.0', port=port, debug=False)
