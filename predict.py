# src/predict.py

import pandas as pd
import numpy as np
import joblib

def load_artifacts():
    artifacts = joblib.load("outputs/model.pkl")
    return artifacts['model'], artifacts['scaler'], artifacts['features']


def predict_transaction(transaction: dict) -> str:
    """
    Predict if a single transaction is fraud.
    
    Example transaction:
    {
        'amount': 500000,
        'oldbalanceOrg': 500000,
        'newbalanceOrig': 0,
        'oldbalanceDest': 0,
        'newbalanceDest': 500000,
        'type_encoded': 1,
        'error_balance_orig': 0,
        'error_balance_dest': 0,
        'orig_drained': 1,
        'amount_ratio': 1.0
    }
    """
    model, scaler, features = load_artifacts()
    
    df = pd.DataFrame([transaction])
    df_scaled = scaler.transform(df[features])
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    result = "FRAUD" if prediction == 1 else "LEGIT"
    print(f"\nPrediction: {result}")
    print(f"Fraud Probability: {probability:.4f}")
    
    return result, probability