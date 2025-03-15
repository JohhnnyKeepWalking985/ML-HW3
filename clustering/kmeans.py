import numpy as np
import pandas as pd
import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class KMeansClustering:
    def __init__(self, n_clusters=3, distance_metric='euclidean', random_state=42, save_data=False):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.save_data = save_data
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.data = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        self.df = X
        self.data = X.iloc[:, :-1].values
        self.labels = self.model.fit_predict(self.data)
        self.inertia_ = self.model.inertia_
        return self.labels

    def silhouette_score(self):
        if self.labels is None:
            raise ValueError("Model has not been fitted yet!")
        return silhouette_score(self.data, self.labels)

    def save_results(self, filename="../datasets/clustering/mkt_kmeans_train.csv"):
        if self.save_data:
            self.df['cluster'] = self.labels
            self.df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    def plot_clusters(self):
        if self.data.shape[1] != 2:
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(self.data)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
            plt.title(f"t-SNE Projection of K-Means Clusters (n_clusters={self.n_clusters})")
            plt.show()
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
            plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Centroids")
            plt.title(f"K-Means Clustering (n_clusters={self.n_clusters})")
            plt.legend()
            plt.show()

    def save_model(self, filename="../model_checkpoints/mkt_kmeans_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def inference(model_path, df, output_file="../datasets/clustering/mkt_kmeans_test.csv"):
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")

        data = df.iloc[:, :-1].values
        labels = model.predict(data)

        df['cluster'] = labels
        df.to_csv(output_file, index=False)
        print(f"Inference results saved to {output_file}")