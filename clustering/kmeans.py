import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class KMeansClustering:
    def __init__(self, n_clusters=3, distance_metric='euclidean', save_data=False):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.save_data = save_data
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.data = None
        self.labels = None

    def fit(self, X):
        """Fit the KMeans model to the data."""
        self.data = X
        self.labels = self.model.fit_predict(X)
        return self.labels

    def silhouette_score(self):
        """Calculate the silhouette score."""
        if self.labels is None:
            raise ValueError("Model has not been fitted yet!")
        return silhouette_score(self.data, self.labels)

    def save_results(self, filename="cluster_results.csv"):
        """Save the cluster results to a CSV file."""
        if self.save_data:
            df = pd.DataFrame(self.data, columns=[f"feature_{i}" for i in range(self.data.shape[1])])
            df['cluster'] = self.labels
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    def plot_clusters(self):
        """Scatter plot of clusters (only for 2D data)."""
        if self.data.shape[1] != 2:
            print("Data is not 2D; cannot plot scatter.")
            return
        
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
        plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Centroids")
        plt.title(f"K-Means Clustering (n_clusters={self.n_clusters})")
        plt.legend()
        plt.show()

