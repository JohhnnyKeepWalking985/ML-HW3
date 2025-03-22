import numpy as np
import joblib
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

class ICADecomposition:
    def __init__(self, n_components=None, random_state=42, save_data=False):
        self.n_components = n_components
        self.save_data = save_data
        self.model = FastICA(n_components=n_components, random_state=random_state)
        self.df = None
        self.data = None
        self.transformed_data = None
        self.reconstruction_error_ = None
        self.kurtosis = None

    def fit(self, X):
        self.df = X
        self.data = X.iloc[:, :-1].values
        self.transformed_data = self.model.fit_transform(self.data)

        reconstructed_data = self.model.inverse_transform(self.transformed_data)
        self.reconstruction_error_ = np.mean((self.data - reconstructed_data) ** 2)
        self.kurtosis = np.mean(np.abs(kurtosis(self.transformed_data, axis=0)))

        return self.transformed_data

    def save_model(self, filename="../model_checkpoints/mkt/ica_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"ICA model saved to {filename}")

    def save_results(self, filename="../datasets/dr/mkt/ica_train.csv"):
        if self.transformed_data is None or self.df is None:
            raise ValueError("Model has not been fitted yet!")

        last_column_name = self.df.columns[-1]
        last_column = self.df[last_column_name]

        transformed_df = pd.DataFrame(self.transformed_data,
                                      columns=[f"IC{i+1}" for i in range(self.transformed_data.shape[1])])
        transformed_df[last_column_name] = last_column.values

        transformed_df.to_csv(filename, index=False)
        print(f"Transformed data saved to {filename}")

    @staticmethod
    def inference(model_filename, df, output_file="../datasets/dr/mkt/ica_test.csv"):
        print(f"Loading ICA model from {model_filename}...")
        model = joblib.load(model_filename)
        print("Model loaded successfully!")

        features = df.iloc[:, :-1].values
        transformed_data = model.transform(features)

        ica_instance = ICADecomposition(n_components=model.n_components)
        ica_instance.df = df
        ica_instance.transformed_data = transformed_data
        ica_instance.save_results(output_file)
