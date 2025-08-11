# ------------------------------------------------------------
# Q3: K-Means Clustering with Visualization
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# 1. Generate sample 2D data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# 2. Apply K-Means clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3. Plot clustered data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("K-Means Clustering Example")
plt.legend()
plt.show()

