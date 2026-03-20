import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import scale_data, apply_pca
from src.clustering.kmeans import run_kmeans
from src.clustering.hierarchical import run_hierarchical
from src.clustering.dbscan import run_dbscan
from src.evaluation import compute_silhouette

# Paths & Folder Setup
data_path = "data/raw/customer_dataset.csv"

os.makedirs("results/pca_outputs", exist_ok=True)
os.makedirs("results/cluster_plots", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# Data Preprocessing
df = preprocess_pipeline(data_path)

# Feature Scaling & PCA
X_scaled = scale_data(df)
X_pca = apply_pca(X_scaled)

# Save PCA outputs
np.save("results/pca_outputs/scaled_data.npy", X_scaled)
np.save("results/pca_outputs/pca_data.npy", X_pca)

# Clustering Models--
kmeans_labels = run_kmeans(X_pca)
agg_labels = run_hierarchical(X_pca)
dbscan_labels = run_dbscan(X_pca)

# Evaluation (Silhouette Score)-----
k_score = compute_silhouette(X_pca, kmeans_labels)
a_score = compute_silhouette(X_pca, agg_labels)

# Fix for DBSCAN (remove noise points)
mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    db_score = compute_silhouette(X_pca[mask], dbscan_labels[mask])
else:
    db_score = 0

# Save metrics
with open("results/metrics/silhouette_scores.txt", "w") as f:
    f.write(f"KMeans: {k_score}\n")
    f.write(f"Hierarchical: {a_score}\n")
    f.write(f"DBSCAN: {db_score}\n")

# Visualization (Cluster Plots)

# KMeans Plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels)
plt.title("KMeans Clusters")
plt.savefig("results/cluster_plots/kmeans.png")
plt.close()

# Hierarchical Plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=agg_labels)
plt.title("Hierarchical Clusters")
plt.savefig("results/cluster_plots/hierarchical.png")
plt.close()

# DBSCAN Plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels)
plt.title("DBSCAN Clusters")
plt.savefig("results/cluster_plots/dbscan.png")
plt.close()

# Save Final Output
df['KMeans_Cluster'] = kmeans_labels
df.to_csv("results/customer_segments.csv", index=False)

# Final Output
print("Silhouette Scores:")
print("KMeans:", k_score)
print("Hierarchical:", a_score)
print("DBSCAN:", db_score)

print("\nCluster Summary:")
print(df.groupby("KMeans_Cluster").mean())

print("\n✅ Project executed successfully!")