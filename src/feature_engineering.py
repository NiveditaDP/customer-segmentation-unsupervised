from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_data(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca