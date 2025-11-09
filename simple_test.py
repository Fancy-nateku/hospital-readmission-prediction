import json
import urllib.request

# Test data
test_data = {
    'age': 65,
    'length_of_stay': 5,
    'num_lab_procedures': 18,
    'num_medications': 12,
    'glucose_level': 110,
    'admission_type': 'Emergency',
    'primary_diagnosis': 'Heart Failure'
}

# Convert to JSON
json_data = json.dumps(test_data).encode('utf-8')

# Make request
req = urllib.request.Request(
    'http://localhost:5000/predict',
    data=json_data,
    headers={'Content-Type': 'application/json'}
)

try:
    response = urllib.request.urlopen(req)
    result = response.read().decode('utf-8')
    print("✅ API Response:", result)
except Exception as e:
    print("❌ Error:", e)