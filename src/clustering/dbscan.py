from sklearn.cluster import DBSCAN

def run_dbscan(X, eps=1.0, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels