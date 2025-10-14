import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


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

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness'] #dropped features with strong correlations
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
    print(score)

kmeans6() #silhouette score improvement after outlier removal
    