import pandas as pd

def preprocess_pipeline(path):
    df = pd.read_csv(path)
    # example: drop NA, simple encoding
    df = df.dropna()
    return df
















