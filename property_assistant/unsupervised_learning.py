
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Illustrative rental listings for learning.
properties = pd.DataFrame({
    "location": [
        "Ruiru", "Juja", "Kiambu", "Ruiru", "Juja",
        "Kiambu", "Ruiru", "Juja", "Kiambu", "Ruiru",
        "Juja", "Kiambu", "Ruiru", "Juja", "Kiambu",
        "Ruiru", "Juja", "Kiambu", "Ruiru", "Juja",
    ],
    "property_type": [
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
    ],
    "bedrooms": [
        1, 3, 2, 4, 2, 3, 1, 4, 2, 3,
        2, 4, 1, 3, 2, 4, 1, 3, 2, 4,
    ],
    "area_sqm": [
        38, 120, 70, 160, 75, 135, 42, 150, 68, 115,
        80, 145, 45, 125, 72, 155, 40, 118, 78, 140,
    ],
    "rent_kes": [
        12000, 65000, 28000, 90000, 32000, 75000, 15000,
        85000, 30000, 60000, 35000, 80000, 16000, 68000,
        29000, 88000, 14000, 62000, 34000, 78000,
    ],
})


# Scale numerical features before distance-based clustering.
numeric_features = ["bedrooms", "area_sqm", "rent_kes"]
X = properties[numeric_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# K-Means: discover three property groups.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
properties["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

print("Property groups:")
print(
    properties[
        ["location", "property_type", "bedrooms",
         "area_sqm", "rent_kes", "kmeans_cluster"]
    ].head(10)
)

print("\nAverage features by group:")
print(
    properties.groupby("kmeans_cluster")[numeric_features].mean()
)

plt.scatter(
    properties["area_sqm"],
    properties["rent_kes"],
    c=properties["kmeans_cluster"],
    cmap="viridis",
)
plt.xlabel("Area (sqm)")
plt.ylabel("Monthly rent (KES)")
plt.title("Property Groups Discovered by K-Means")
plt.show()


# Elbow method: compare different cluster counts.
inertias = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.plot(range(1, 8), inertias, "bo-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Property Listings")
plt.show()


# PCA: reduce three features to two components.
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(
    "\nVariance retained by two PCA components:",
    f"{pca.explained_variance_ratio_.sum():.1%}",
)

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=properties["kmeans_cluster"],
    cmap="viridis",
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Property Features Reduced with PCA")
plt.show()


# DBSCAN: find dense groups and potential outliers.
dbscan = DBSCAN(eps=0.9, min_samples=3)
properties["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

n_clusters = properties.loc[
    properties["dbscan_cluster"] != -1,
    "dbscan_cluster",
].nunique()

n_noise = (properties["dbscan_cluster"] == -1).sum()

print(f"\nDBSCAN clusters: {n_clusters}")
print(f"Potential outliers: {n_noise}")

plt.scatter(
    properties["area_sqm"],
    properties["rent_kes"],
    c=properties["dbscan_cluster"],
    cmap="plasma",
)
plt.xlabel("Area (sqm)")
plt.ylabel("Monthly rent (KES)")
plt.title("DBSCAN Property Clusters (-1 marks noise)")
plt.show()