import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('../data/spotify_clean.csv')

def kmeans1():
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

    """
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
    """
    clusters = 5

    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)

    df['label'] = labels
    print(df.groupby('label')[features].mean())

    print(silhouette_score(X_reduced, kmeans.labels_))


def kmeans2():
    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'acousticness', 'speechiness', 'loudness', 'instrumentalness']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    comps = range(pca.n_components_)

    plt.bar(comps, pca.explained_variance_)
    plt.xlabel('PCA Feature')
    plt.ylabel('Variance')
    #plt.show()

    pca = PCA(n_components=6)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    inertias = []

    for k in range(2, 11):

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_reduced)
        inertia = kmeans.inertia_
        inertias.append(inertia)


    ks = range(2, 11)
    plt.plot(ks, inertias, marker = 'o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    """
    silhouette_scores = []

    for k in range(2, 11):

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_reduced)
        score = silhouette_score(X_reduced, kmeans.labels_)
        silhouette_scores.append(score)

    ks = range(2, 11)
    plt.plot(ks, silhouette_scores, marker = 'o')
    plt.xlabel('Number of clusters' )
    plt.ylabel('Silhouette score')
    plt.title('Silhouette method')
    plt.show()
    """
    clusters = 5

    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)

    df['label'] = labels
    print(df.groupby('label')[features].mean())

    print(silhouette_score(X_reduced, kmeans.labels_))

silhouette_scores = []
features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'acousticness', 'speechiness', 'loudness', 'instrumentalness']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""
for n_comp in range(3, 9):

    pca = PCA(n_components=n_comp)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    silhouette_scores.append(score)

for idx, score in enumerate(silhouette_scores):
    print(f"No. of PCA components = {idx+3}, Silhouette score = {score}") #3 PCA components had the best silhouette score
"""
def kmeans3():

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'acousticness', 'speechiness', 'loudness', 'instrumentalness']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)
    df['label'] = labels
    score = silhouette_score(X_reduced, labels)

    print(score)

#kmeans3()

pca = PCA()
pca.fit(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cum_var > 0.9) + 1 #np.argmax returns the index of the first cumvar > 0.9
print(n_components)

def kmeans4():

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'acousticness', 'speechiness', 'loudness', 'instrumentalness']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=7)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)
    df['label'] = labels
    score = silhouette_score(X_reduced, labels)
    print(score)

#kmeans4()

def kmeans5():

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness'] #dropped features with strong correlations
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for n_comps in range(3, 7):

        pca = PCA(n_components=n_comps)
        pca.fit(X_scaled)
        X_reduced = pca.transform(X_scaled)

        for n_clusters in range(2, 11):

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_reduced)
            labels = kmeans.predict(X_reduced)
            score = silhouette_score(X_reduced, labels)
            print(f"No. of PCA components: {n_comps}, No. of clusters: {n_clusters}, Silhouette score: {score}")

#kmeans5() #optimal configuration is n_components = 3, n_clusters = 4


def remove_outliers(X, df):

    Q1 = np.quantile(X, 0.25, axis = 0)
    Q3 = np.quantile(X, 0.75, axis = 0)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR

    mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis = 1)
    return X[mask], df[mask].copy()

def kmeans6():

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness']#dropped features with strong correlations
    print(df.head()) 
    X = df[features]

    X_clean, df_clean = remove_outliers(X, df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    pca = PCA(n_components = 3)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    print(f"Silhouette score: {score}")
    
    print(pd.Series(labels).value_counts().sort_index())
    

#kmeans6() #silhouette score improvement after outlier removal

def dbscan_pipeline(X, df, use_pca = True, n_components = 3):

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    if use_pca:

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        X_processed = pca.transform(X_scaled)
    
    else:
        X_processed = X

    k = 5
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_processed)
    distances, _ = neighbors.kneighbors(X_processed)
    distances = np.sort(distances[:, k-1])

    suggested_eps = np.percentile(distances, 90)
    print(f'Suggested eps (90th percentile): {suggested_eps}')

    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f'Suggested eps={suggested_eps:.2f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.legend()
    plt.grid(True)
    plt.show()

    dbscan = DBSCAN(eps=suggested_eps, min_samples=k)
    dbscan.fit(X_processed)
    labels = dbscan.fit_predict(X_processed)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nResults:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    if n_clusters > 1:

        mask = labels != -1
        if sum(mask) > 0:

            score = silhouette_score(X_processed[mask], labels[mask])
            print(f"Silhouette score: {score}")


features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness'] #dropped features with strong correlations
X = df[features]
df = pd.read_csv('../data/spotify_clean.csv', index_col=0)

#dbscan_pipeline(X, df,  use_pca=True, n_components=3)

kmeans6()

def different_clustering(X, df, use_pca = True, n_components = 3):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        X_processed = pca.transform(X_scaled)
    
    else:

        X_processed = X

    algs = {'KMeans': KMeans(n_clusters = 4, random_state=42), 'Hierarchical': AgglomerativeClustering(n_clusters=4), 'Spectral': SpectralClustering(n_clusters=4, random_state=42)}

    for name, algo in algs.items():

        if hasattr(algo, 'predict'):
            algo.fit(X_processed)
            labels = algo.predict(X_processed)
        else:
            labels = algo.fit_predict(X_processed)
            
        score = silhouette_score(X_processed, labels)
        print(f"{name}: {score}")

different_clustering(X, df, use_pca=True, n_components=3)