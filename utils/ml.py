import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('../data/spotify_clean.csv')

features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'speechiness', 'loudness', 'instrumentalness']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform (X)


pca = PCA()
pca.fit(X_scaled)

comps = range(pca.n_components_)
plt.bar(comps, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('Variance')
#plt.show()

plt.plot(np.cumsum((pca.explained_variance_ratio_)))
plt.xlabel('PCA feature')
plt.ylabel('Cumulative explained variance')
#plt.show()

pca = PCA(n_components=6)
X_reduced = pca.fit_transform(X_scaled)

inertias = []
for k in range(2, 11):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    inertias.append(kmeans.inertia_)

ks = range(2, 11)
plt.plot(ks, inertias, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

silhouette_scores = []
for k in range(2, 11):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    score = silhouette_score(X_reduced, kmeans.labels_)
    silhouette_scores.append(score)

ks = range(2, 11)
plt.plot(ks, silhouette_scores, marker = 'o')
plt.xlabel('Number of clusters' )
plt.ylabel('Silhouette score')
plt.title('Silhouette method')
plt.show()

clusters = 5