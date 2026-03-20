from sklearn.mixture import GaussianMixture

def run_gmm(X_scaled):
    model = GaussianMixture(n_components=2, random_state=42)
    return model.fit_predict(X_scaled)