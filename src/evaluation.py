from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_silhouette(X, labels):
    """
    Compute Silhouette Score
    """
    # If only one cluster, silhouette is invalid
    if len(set(labels)) <= 1:
        return 0
    return silhouette_score(X, labels)


def compute_davies_bouldin(X, labels):
    """
    Compute Davies-Bouldin Index
    """
    if len(set(labels)) <= 1:
        return 0
    return davies_bouldin_score(X, labels)