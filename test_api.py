import requests
import json
import pandas as pd

# Sample test data matching your model's expected features
test_data = {
    'age': 65,
    'length_of_stay': 5,
    'num_lab_procedures': 18,
    'num_medications': 12,
    'glucose_level': 110,
    'admission_type': 'Emergency',
    'primary_diagnosis': 'Heart Failure'
}

try:
    # Test the prediction endpoint
    response = requests.post('http://localhost:5000/predict', json=test_data)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
    
    # Test health endpoint
    health_response = requests.get('http://localhost:5000/health')
    print("\nHealth Check:", health_response.json())
    
except Exception as e:
    print("Error testing API:", e)
    print("\nTo test locally, run: python app.py")
    print("Then in another terminal run: python test_api.py")
