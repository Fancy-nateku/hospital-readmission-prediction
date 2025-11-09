import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Create sample data to match your original structure
np.random.seed(42)
n = 100  # Small sample just to recreate the preprocessor

data = {
    'age': np.random.randint(18, 95, n),
    'length_of_stay': np.random.exponential(5, n).astype(int) + 1,
    'num_lab_procedures': np.random.poisson(20, n),
    'num_medications': np.random.poisson(15, n),
    'glucose_level': np.random.normal(120, 30, n),
    'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n),
    'primary_diagnosis': np.random.choice(['Heart Failure', 'Pneumonia', 'Diabetes', 'COPD'], n),
    'readmitted': np.random.binomial(1, 0.25, n)
}

df = pd.DataFrame(data)
X = df.drop('readmitted', axis=1)

# Recreate the preprocessor exactly as before
cat_cols = ['admission_type', 'primary_diagnosis']
num_cols = ['age', 'length_of_stay', 'num_lab_procedures', 'num_medications', 'glucose_level']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Fit and save
preprocessor.fit(X)
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor recreated successfully!")