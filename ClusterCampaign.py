import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load dataset
dataset = pd.read_csv('insurance.csv')

 
print(dataset.isnull().sum())
dataset = dataset.dropna()

dataset = dataset.drop(['charges'], axis=1)

dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)

#Scaling & Normalization
from sklearn.preprocessing import StandardScaler, normalize

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# Normalize
normalized_data = normalize(scaled_data)

X = normalized_data

from sklearn.cluster import KMeans

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    
    print(f"k = {k}, inertia = {inertia}")

#Elbow curve
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#Pptimal k
optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

print("K-Means clusters (first 10):", kmeans_labels[:10])

from sklearn.cluster import MeanShift

#Different bandwidths
bandwidths = [0.5, 1, 2, 3]

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw)
    ms.fit(X)
    
    labels = ms.labels_
    n_clusters = len(np.unique(labels))
    
    print(f"Bandwidth = {bw}, Clusters = {n_clusters}")