import numpy as np
import joblib
import pandas as pd
from sklearn.decomposition import PCA

class PCADecomposition:
    def __init__(self, n_components=None, random_state=42, save_data=False):
        self.n_components = n_components
        self.save_data = save_data
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None
        self.df = None
        self.data = None
        self.transformed_data = None

    def fit(self, X):
        self.df = X
        self.data = X.iloc[:, :-1].values
        self.transformed_data = self.model.fit_transform(self.data)
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        return self.transformed_data

    def save_model(self, filename="../model_checkpoints/mkt/pca_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"PCA model saved to {filename}")

    def save_results(self, filename="../datasets/dr/mkt/pca_train.csv"):
        if self.transformed_data is None or self.df is None:
            raise ValueError("Model has not been fitted yet!")

        last_column_name = self.df.columns[-1]
        last_column = self.df[last_column_name]

        transformed_df = pd.DataFrame(self.transformed_data,
                                      columns=[f"PC{i+1}" for i in range(self.transformed_data.shape[1])])
        transformed_df[last_column_name] = last_column.values

        transformed_df.to_csv(filename, index=False)
        print(f"Transformed data saved to {filename}")

    @staticmethod
    def inference(model_filename, df, output_file="../datasets/dr/mkt/pca_test.csv"):
        print(f"Loading PCA model from {model_filename}...")
        model = joblib.load(model_filename)
        print("Model loaded successfully!")

        features = df.iloc[:, :-1].values
        transformed_data = model.transform(features)

        pca_instance = PCADecomposition(n_components=model.n_components)
        pca_instance.df = df
        pca_instance.transformed_data = transformed_data
        pca_instance.save_results(output_file)
