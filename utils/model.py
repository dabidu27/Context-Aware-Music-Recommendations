from clean_lyrics import clean_lyrics
import pandas as pd
from sentence_transformers import InputExample, losses, SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

def remove_outliers(X, df):

    Q1 = np.quantile(X, 0.25, axis = 0)
    Q3 = np.quantile(X, 0.75, axis = 0)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR

    mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis = 1)
    return X[mask], df[mask].copy()


def kmeans_model(df):

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness']#dropped features with strong correlations
    print(df.head()) 
    X = df[features]

    X_clean, df_clean = remove_outliers(X, df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    df_clean['audio_embedding'] = list(X_scaled)

    pca = PCA(n_components = 3)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_reduced)
    labels = kmeans.predict(X_reduced)

    df_clean['label'] = labels

    return df_clean

def create_pairs(df):
    pairs = []
    for i, row in df.iterrows():

        similar = df[df['label'] == row['label']].sample(1).iloc[0]
        different = df[df['label'] != row['label']].sample(1).iloc[0]

        pairs.append([row['clean_lyrics'], similar['clean_lyrics'], 1.0])
        pairs.append([row['clean_lyrics'], different['clean_lyrics'], 0.0])

    return pd.DataFrame(pairs, columns=['lyric1', 'lyric2', 'label'])


def finetune_bert(df, model_name = 'all-MiniLM-L6-v2', num_epochs = 3):

    model = SentenceTransformer(model_name)

    train_examples = [InputExample(texts=[t1, t2], label=float(label)) for t1, t2, label in zip(df['lyric1'], df['lyric2'], df['label'])]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    trainloss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives = [(train_dataloader, trainloss)], epochs=num_epochs, warmup_steps=100)

    return model

def create_lyrics_embedding(df, model):

    df['lyrics_embedding'] = df['clean_lyrics'].apply(lambda x: model.encode(x))
    return df

def recommend_by_lyrics(df):

    song_name = df.sample(1).iloc[0]['track_name']
    print(f'Similar songs to {song_name}')
    query_vec = df.loc[df['track_name'] == song_name, 'lyrics_embedding'].values[0]
    lyrics_matrix = np.stack(df['lyrics_embedding'].values)
    similarities = cosine_similarity([query_vec], lyrics_matrix)[0]

    df['similarities'] = similarities
    recommendations = df.sort_values('similarities', ascending = False).head(6).iloc[1:][['track_name', 'track_artist', 'similarities']]
    return recommendations

def recommend_by_query(df, query, model):
    
    print(query)
    query_vec = model.encode(query)
    lyrics_matrix = np.stack(df['lyrics_embedding'].values)
    similarities = cosine_similarity([query_vec], lyrics_matrix)[0]

    df['similarities'] = similarities
    recommendations = df.sort_values('similarities', ascending = False).head(5)[['track_name', 'track_artist', 'similarities']]
    return recommendations


if __name__ == "__main__":

    df = pd.read_csv('../data/sample1000_with_lyrics.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~((df['instrumentalness'] < 0.9) & (df['lyrics'].isna()))].reset_index(drop = True)

    df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)
    #print(df[['track_name', 'clean_lyrics']].head())

    df = kmeans_model(df)
    pairs_df = create_pairs(df)
    #model = finetune_bert(pairs_df, model_name = 'all-MiniLM-L6-v2', num_epochs = 3)
    #model.save('../models/finetuned_bert')

    model = SentenceTransformer('../models/finetuned_bert')
    df = create_lyrics_embedding(df, model)
    #print(df.head())

    """
    s = 0
    for i in range(10):
        s1 = df[df['label'] == 1].sample(1).iloc[0]['lyrics_embedding']
        s2 = df[df['label'] == 1].sample(1).iloc[0]['lyrics_embedding']
        similarity = cosine_similarity([s1], [s2])[0][0]
        print(similarity)

    """

    recommendations = recommend_by_query(df, 'Hey! I want some rock songs', model)
    print(recommendations)