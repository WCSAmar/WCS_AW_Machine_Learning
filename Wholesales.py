import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

dataset = pd.read_csv('Wholesale customers data.csv')
#print(dataset.head())
print(dataset)

dataset = dataset.dropna()


categorical = dataset[['Channel', 'Region']]

encoder = OneHotEncoder()
encoded_cat = encoder.fit_transform(categorical).toarray()

numerical = dataset.drop(['Channel', 'Region'], axis=1)


scaler = StandardScaler()
scaled_num = scaler.fit_transform(numerical)

X = np.concatenate([scaled_num, encoded_cat], axis=1)

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    
    print(f"k = {k}, inertia = {inertia}")

    import matplotlib.pyplot as plt

plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

best_k = 4
print("Best k (from elbow):", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
print(clusters[:10])