from sklearn.metrics import silhouette_score

def evaluate_model(X_scaled, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X_scaled, labels)
    else:
        return "Invalid clustering"