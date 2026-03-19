import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import scale_data, apply_pca
from src.clustering.kmeans import run_kmeans
from src.clustering.hierarchical import run_hierarchical
from src.clustering.dbscan import run_dbscan
from src.evaluation import compute_silhouette
import os

# paths
data_path = "data/raw/customer_dataset.csv"
os.makedirs("results/pca_outputs", exist_ok=True)
os.makedirs("results/cluster_plots", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# preprocessing
df = preprocess_pipeline(data_path)
X_scaled = scale_data(df)
X_pca = apply_pca(X_scaled)

# save
np.save("results/pca_outputs/scaled_data.npy", X_scaled)
np.save("results/pca_outputs/pca_data.npy", X_pca)

# clustering
kmeans_labels = run_kmeans(X_pca)
agg_labels = run_hierarchical(X_pca)
dbscan_labels = run_dbscan(X_pca)

# silhouette scores
k_score = compute_silhouette(X_pca, kmeans_labels)
a_score = compute_silhouette(X_pca, agg_labels)
# DBSCAN may have -1 labels
db_score = compute_silhouette(X_pca, dbscan_labels[dbscan_labels!=-1]) if len(set(dbscan_labels))>1 else 0

# save metrics
with open("results/metrics/silhouette_scores.txt", "w") as f:
    f.write(f"KMeans: {k_score}\nHierarchical: {a_score}\nDBSCAN: {db_score}")

# save cluster plots
import matplotlib.pyplot as plt

plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels)
plt.title("KMeans Clusters")
plt.savefig("results/cluster_plots/kmeans.png")
plt.close()