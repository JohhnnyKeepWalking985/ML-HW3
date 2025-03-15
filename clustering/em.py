import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

class EMClustering:
    def __init__(self, 
                 n_components=3, 
                 covariance_type='full', 
                 max_iter=200, 
                 tol=1e-4, 
                 reg_covar=1e-6, 
                 init_params='kmeans', 
                 random_state=42, 
                 save_data=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.save_data = save_data
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            init_params=init_params,
            random_state=random_state
        )
        
        self.data = None
        self.labels = None
        self.df = None
        self.score = None

    def fit(self, X):
        self.df = X
        self.data = X.iloc[:, :-1].values
        self.model.fit(self.data)
        self.labels = self.model.predict(self.data)
        self.score = -1 * self.model.score(self.data)
        return self.labels

    def silhouette_score(self):
        if self.labels is None:
            raise ValueError("Model has not been fitted yet!")
        return silhouette_score(self.data, self.labels)

    def save_results(self, filename="../datasets/clustering/mkt_em_train.csv"):
        if self.save_data:
            self.df['cluster'] = self.labels
            self.df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    def plot_clusters(self):
        if self.data.shape[1] != 2:
            tsne = TSNE(n_components=2, random_state=self.random_state)
            X_tsne = tsne.fit_transform(self.data)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
            plt.title(f"t-SNE Projection of EM Clusters (n_components={self.n_components})")
            plt.show()
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
            plt.scatter(self.model.means_[:, 0], self.model.means_[:, 1], c='red', marker='X', s=200, label="Cluster Means")
            plt.title(f"EM Clustering (n_components={self.n_components})")
            plt.legend()
            plt.show()

    def save_model(self, filename="../model_checkpoints/mkt_em_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def inference(model_path, df, output_file="../datasets/clustering/mkt_em_test.csv"):
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")

        data = df.iloc[:, :-1].values
        labels = model.predict(data)

        df['cluster'] = labels
        df.to_csv(output_file, index=False)
        print(f"Inference results saved to {output_file}")
