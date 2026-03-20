import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_pipeline(path):
    df = load_data(path)
    
    # handle missing values
    df = df.dropna()
    
    return df