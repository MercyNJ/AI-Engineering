
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

# Scale features before distance-based clustering and PCA.
X_scaled = StandardScaler().fit_transform(X)

# Reduce four features to two for visualisation.
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Discover three clusters using all four scaled features.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Project the centroids into the same 2D PCA space.
centroids_2d = pca.transform(kmeans.cluster_centers_)

plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=labels, cmap="viridis", alpha=0.8,
)
plt.scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    s=200, c="red", marker="X", label="Centroids",
)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("K-Means on Iris (PCA View)")
plt.legend()
plt.show()