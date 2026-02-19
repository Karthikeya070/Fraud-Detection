import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/plots", exist_ok=True)

def class_distribution(df: pd.DataFrame):
    """
    Check how imbalanced the dataset is.
    """
    count = df['isFraud'].value_counts()
    percent = df['isFraud'].value_counts(normalize=True) * 100
    
    print("\n--- Class Distribution ---")
    print(f"Non-Fraud (0): {count[0]} ({percent[0]:.4f}%)")
    print(f"Fraud (1):     {count[1]} ({percent[1]:.4f}%)")
    
    # Plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='isFraud', data=df, palette=['steelblue','tomato'])
    plt.title("Class Distribution (0=Legit, 1=Fraud)")
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.xticks([0,1], ['Legit','Fraud'])
    plt.savefig("outputs/plots/class_distribution.png")
    plt.close()
    print("Saved: class_distribution.png")


def transaction_type_analysis(df: pd.DataFrame):
    """
    Fraud only happens in TRANSFER and CASH_OUT types.
    """
    print("\n--- Fraud by Transaction Type ---")
    print(df.groupby('type')['isFraud'].sum())
    
    plt.figure(figsize=(8,4))
    sns.countplot(x='type', hue='isFraud', data=df, palette=['steelblue','tomato'])
    plt.title("Transaction Type vs Fraud")
    plt.savefig("outputs/plots/transaction_type.png")
    plt.close()
    print("Saved: transaction_type.png")


def amount_distribution(df: pd.DataFrame):
    """
    Distribution of transaction amounts for fraud vs legit.
    """
    plt.figure(figsize=(10,4))
    
    df[df['isFraud']==0]['amount'].apply(lambda x: min(x, 1e6)).hist(
        bins=50, alpha=0.6, label='Legit', color='steelblue'
    )
    df[df['isFraud']==1]['amount'].apply(lambda x: min(x, 1e6)).hist(
        bins=50, alpha=0.6, label='Fraud', color='tomato'
    )
    
    plt.title("Amount Distribution — Fraud vs Legit")
    plt.xlabel("Amount (capped at 1M)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("outputs/plots/amount_distribution.png")
    plt.close()
    print("Saved: amount_distribution.png")


def correlation_heatmap(df: pd.DataFrame):
    """
    Correlation between numerical features.
    """
    numeric_df = df.select_dtypes(include='number')
    
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.close()
    print("Saved: correlation_heatmap.png")


def run_eda(df: pd.DataFrame):
    class_distribution(df)
    transaction_type_analysis(df)
    amount_distribution(df)
    correlation_heatmap(df)