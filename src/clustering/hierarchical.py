from sklearn.cluster import AgglomerativeClustering

def run_hierarchical(X, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels