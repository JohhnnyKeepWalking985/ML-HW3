import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('datasets/raw_data/marketing_campaign.csv')

numeric_data = data.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

em = GaussianMixture(n_components=5, random_state=42)
em.fit(scaled_data)

data['Cluster'] = em.predict(scaled_data)
data.to_csv('datasets/clustering/mkt_campaign_em.csv', index=False)
