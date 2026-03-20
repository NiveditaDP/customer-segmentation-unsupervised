def feature_engineering(df):
    # example: create new feature
    df['Income_per_Product'] = df['Income Level'] / (df['Insurance Products Owned'] + 1)
    
    return df