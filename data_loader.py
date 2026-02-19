import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    return df