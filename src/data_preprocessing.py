# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(filepath)
    return df


def basic_cleaning(df):
    """
    Perform basic cleaning:
    - Drop Customer ID
    - Handle missing values
    """
    df = df.copy()

    # Drop Customer ID if exists
    if "Customer ID" in df.columns:
        df = df.drop("Customer ID", axis=1)

    # Drop rows with missing values
    df = df.dropna()

    return df


def encode_categorical(df):
    """
    Encode categorical columns using Label Encoding
    """
    df = df.copy()
    le = LabelEncoder()

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def scale_features(df):
    """
    Scale numerical features using StandardScaler
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)

    return scaled_df


def preprocess_pipeline(filepath):
    """
    Complete preprocessing pipeline
    """
    df = load_data(filepath)
    df = basic_cleaning(df)
    df_encoded = encode_categorical(df)
    df_scaled = scale_features(df_encoded)

    return df_scaled