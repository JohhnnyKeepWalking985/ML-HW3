import numpy as np
import joblib
import pandas as pd
import umap 

class UMAPDecomposition:
    def __init__(self, n_components=None, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, save_data=False):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.save_data = save_data
        self.model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
        self.df = None
        self.data = None
        self.transformed_data = None

    def fit(self, X):
        self.df = X
        self.data = X.iloc[:, :-1].values
        self.transformed_data = self.model.fit_transform(self.data)
        return self.transformed_data

    def save_model(self, filename="../model_checkpoints/mkt/umap_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"UMAP model saved to {filename}")

    def save_results(self, filename="../datasets/dr/mkt/umap_train.csv"):
        if self.transformed_data is None or self.df is None:
            raise ValueError("Model has not been fitted yet!")

        last_column_name = self.df.columns[-1]
        last_column = self.df[last_column_name]

        transformed_df = pd.DataFrame(self.transformed_data,
                                      columns=[f"UMAP{i+1}" for i in range(self.transformed_data.shape[1])])
        transformed_df[last_column_name] = last_column.values

        transformed_df.to_csv(filename, index=False)
        print(f"Transformed data saved to {filename}")

    @staticmethod
    def inference(model_filename, df, output_file="../datasets/dr/mkt/umap_test.csv"):
        print(f"Loading UMAP model from {model_filename}...")
        model = joblib.load(model_filename)
        print("Model loaded successfully!")

        features = df.iloc[:, :-1].values
        transformed_data = model.transform(features)

        umap_instance = UMAPDecomposition(n_components=model.n_components)
        umap_instance.df = df
        umap_instance.transformed_data = transformed_data
        umap_instance.save_results(output_file)
