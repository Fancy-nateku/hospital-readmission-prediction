Predicting 30-Day Hospital Readmission Risk Using XGBoost

Prepared by: Marcus Garvey Kinyenye, Fancy Nateku

Institution: PLP Academy

Date: Sunday, November 9, 2025



Table of Contents

Executive Summary

Project Overview

Objectives

Data

Model Development

Actual Model Performance

Performance Gap Analysis

Deployment

Model Explainability

Ethics and Bias Considerations

Lessons Learned

AI Development Workflow

Setup & Usage

References

Executive Summary

This project demonstrates a complete AI development workflow for predicting 30-day hospital readmissions. While synthetic data limitations affected final performance (AUC: 0.580), the implementation successfully showcases end-to-end MLOps, including model training, explainability with SHAP, API development with Flask, and containerization with Docker.

This is a proof-of-concept using synthetic data, which explains the performance gap from real-world expectations.

Project Overview

The project applies the AI development workflow to a healthcare scenario: predicting hospital readmission risk within 30 days. The pipeline includes data preprocessing, XGBoost model development, evaluation, deployment via Flask API, and Docker containerization.

Objectives

Reduce hospital readmission rates by at least 15%

Enable clinicians to identify high-risk patients

Optimize healthcare resource allocation and reduce operational costs

KPI target: AUC-ROC ≥ 0.80

Stakeholders:

Hospital Administrators

Clinicians and Nurses

Patients

Data

Sources:

Electronic Health Records (EHR): demographics, vitals, lab tests, diagnoses, medications, discharge summaries

Administrative/Claims Data: admission type, hospital stay duration, insurance type, readmission history

Preprocessing Steps:

Handle missing values (median for numeric, mode for categorical)

Manual encoding of categorical variables

Feature scaling using custom standardization

Outlier removal (e.g., glucose > 500)

Derived feature creation (e.g., risk_score)

Potential Bias:

Selection bias due to incomplete records

Algorithmic bias may underpredict risks for certain groups

Model Development

Algorithm: XGBoost Classifier

Justification:

Handles categorical and numerical features effectively

Built-in regularization mitigates overfitting

Supports SHAP explainability

High performance in structured medical datasets

Data Split:

70% Training | 15% Validation | 15% Testing

Hyperparameters Tuned:

max_depth, learning_rate, n_estimators

Regularization parameters: lambda, alpha

Actual Model Performance

AUC-ROC: 0.580

Confusion Matrix:
| Actual / Predicted | Readmit | No Readmit |
|------------------|---------|------------|
| Readmit | 13 | 64 |
| No Readmit | 28 | 195 |

Recall: 16.9%

Precision: 31.7%

Performance Gap Analysis

The actual model performed below the target (AUC: 0.80) due to:

Synthetic Data Limitations: randomly generated data lacks real clinical patterns

Small Dataset: 2,000 samples vs. thousands needed for robust medical models

Feature Engineering: real EHR would include more predictive features (comorbidities, vital trends, social determinants)

Deployment

Implementation:

Flask API with /predict, /health, and /features endpoints

Docker containerization for portability

Model: XGBoost with base_score=0.5 to avoid conversion issues

Preprocessing: ColumnTransformer with StandardScaler and OneHotEncoder

Regulatory Compliance (HIPAA):

Encrypt PHI at rest and in transit

Restrict model access via authentication and audit logging

Perform annual security and fairness audits

Model Explainability

SHAP Analysis Identified Top Risk Factors:

[Top feature from SHAP plot]

[Second most important feature]

[Third most important feature]

Enables clinicians to understand model decisions and build trust.

Ethics and Bias Considerations

Impact of Biased Data: underrepresentation of groups may worsen outcomes

Mitigation Strategy:

Reweight or oversample minority groups

Perform bias audits using subgroup metrics

Lessons Learned

Synthetic data cannot fully replicate complex medical patterns

Model explainability (SHAP) is crucial for healthcare adoption

Production deployment requires careful handling of dependencies, security, and scalability

AI Development Workflow

Problem Definition

Data Collection

Custom Preprocessing (encoding, scaling, imputation)

Model Development (XGBoost)

Evaluation (AUC, Recall)

Deployment (Flask + Docker)

Monitoring & Maintenance (concept drift detection)

(Insert workflow diagram PNG here — label: “AI Development Workflow for Hospital Readmission Prediction”)

Setup & Usage

Requirements:

Python 3.9+

pip or conda

Docker (optional)

Installation:

git clone [your-repo-url]
cd [repo-folder]
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt


Run Flask API:

python app.py


Endpoints: /predict, /health, /features

Using Docker:

docker build -t hospital-readmission-api .
docker run -p 5000:5000 hospital-readmission-api


Access API at http://localhost:5000

References

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). Advances in Neural Information Processing Systems

U.S. Department of Health & Human Services. (2023). HIPAA Privacy and Security Rules

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier. KDD 2016
