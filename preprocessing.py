

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key insight from EDA:
    - Fraud only happens in TRANSFER and CASH_OUT
    - Balance discrepancies are strong fraud signals
    """
    print("\nEngineering features...")
    
    # 1. Filter only TRANSFER and CASH_OUT — fraud only occurs here
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    # 2. Encode transaction type
    df['type_encoded'] = (df['type'] == 'TRANSFER').astype(int)
    # TRANSFER = 1, CASH_OUT = 0
    
    # 3. Balance error features
    # After transaction, actual balance vs expected balance
    df['error_balance_orig'] = (
        df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])
    )
    df['error_balance_dest'] = (
        df['newbalanceDest'] - (df['oldbalanceDest'] + df['amount'])
    )
    
    # 4. Flag if origin account is completely drained
    df['orig_drained'] = (df['newbalanceOrig'] == 0).astype(int)
    
    # 5. Transaction amount relative to origin balance
    df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    # +1 to avoid division by zero
    
    print(f"Shape after filtering: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame):
    """
    Select features, split, scale, handle imbalance with SMOTE.
    """
    
    # Feature selection
    features = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig', 
        'oldbalanceDest',
        'newbalanceDest',
        'type_encoded',
        'error_balance_orig',
        'error_balance_dest',
        'orig_drained',
        'amount_ratio'
    ]
    
    X = df[features]
    y = df['isFraud']
    
    print(f"\nFeatures used: {features}")
    print(f"X shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    
    # Train-test split (80-20, stratified to maintain class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # Important for imbalanced data
    )
    
    print(f"\nTrain size: {X_train.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    # Note: fit ONLY on train, transform both
    
    # Handle class imbalance with SMOTE on training data only
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )
    
    print(f"After SMOTE — Train shape: {X_train_resampled.shape}")
    print(f"After SMOTE — Class distribution:\n{pd.Series(y_train_resampled).value_counts()}")
    
    return (
        X_train_resampled, X_test_scaled,
        y_train_resampled, y_test,
        scaler, features
    )