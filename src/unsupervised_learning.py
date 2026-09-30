import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../data/processed/customer_360_features.csv')

segmentation_features = [
    "TenureMonths",
    "MonthlyUsageGB",
    "MonthlyChargeEGP",
    "NumServices",
    "SupportCalls6M",
    "LatePayments12M",
    "SatisfactionScore",
    "SupportSatisfactionInteraction",
    "HighValueCustomer",
    "LatePaymentRate",
    "SupportCallsPerTenure",
    "UsagePerService",
    "ChargePerService"
]

segmentation_data = df[segmentation_features].copy()

segmentation_data = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(segmentation_data),
    columns=segmentation_features
)

segmentation_scaled = StandardScaler().fit_transform(segmentation_data)

segmentation_scaled_df = pd.DataFrame(
    segmentation_scaled, columns=segmentation_features)

print("Segmentation matrix:", segmentation_scaled.shape)


k_values = range(2, 11)
inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(segmentation_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(segmentation_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_values), inertias, marker="o")
axes[0].set_title("K-Means Elbow Curve")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_values), silhouettes, marker="o")
axes[1].set_title("K-Means Silhouette Scores")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()

plt.show()

silhouette_df = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertias,
    "Silhouette": silhouettes
})

silhouette_df

best_k = int(silhouette_df.loc[silhouette_df["Silhouette"].idxmax(), "k"])

print("Selected k based on highest silhouette score:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(segmentation_scaled)

df["KMeansCluster"] = kmeans_labels


cluster_profile = df.groupby("KMeansCluster")[
    segmentation_features].mean().round(2)
cluster_sizes = df["KMeansCluster"].value_counts().sort_index()

cluster_sizes.rename("Customers").to_frame()
cluster_profile


pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(segmentation_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_.round(2))
print("Total explained variance:", pca.explained_variance_ratio_.sum().round(2))

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=kmeans_labels,
    alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments in PCA Space")

plt.colorbar(scatter, label="K-Means Cluster")

plt.show()

agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(segmentation_scaled)

agg_silhouette = silhouette_score(segmentation_scaled, agg_labels)

print("Agglomerative silhouette score:", round(agg_silhouette, 4))

min_samples = 10

neighbors = NearestNeighbors(n_neighbors=min_samples)

neighbors.fit(segmentation_scaled)

distances, indices = neighbors.kneighbors(segmentation_scaled)

k_distances = distances[:, -1]
k_distances = np.sort(k_distances)

plt.figure(figsize=(10, 5))

plt.plot(k_distances)

plt.xlabel("Samples sorted by distance")
plt.ylabel(f"Distance to {min_samples}th nearest neighbor")
plt.title("DBSCAN K-Distance Plot", fontweight='bold')

plt.grid(True)

plt.show()

eps_value = 2.5

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)

dbscan_labels = dbscan.fit_predict(segmentation_scaled)

noise_mask = (dbscan_labels == -1)
non_noise_mask = (dbscan_labels != -1)

n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

n_noise = np.sum(noise_mask)

print("Number of clusters:", n_dbscan_clusters)
print("Number of noise points:", n_noise)

plt.figure(figsize=(10, 7))

unique_labels = sorted(np.unique(dbscan_labels))

for label in unique_labels:
    mask = (dbscan_labels == label)

    if label == -1:
        plt.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            marker="x",
            s=50,
            label="Noise"
        )

    else:
        plt.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            s=50,
            alpha=0.7,
            label=f"Cluster {label}"
        )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering on PCA Projection")

plt.legend()
plt.tight_layout()

plt.show()
