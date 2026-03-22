import pandas as pd

def preprocess_pipeline(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Drop Customer ID (not useful)
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)

    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df