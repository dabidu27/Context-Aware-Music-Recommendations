from clean_lyrics import clean_lyrics
import pandas as pd
from sentence_transformers import InputExample, losses, SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity


def kmeans_model(df):

    features = ['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness']

    X = df[features]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    scores = []

    for n_clusters in range (3, 11):

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((n_clusters, score))
    
    best_n_clusters, best_score = max(scores, key = lambda x: x[1])

    final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    final_kmeans.fit(X_scaled)
    labels = final_kmeans.predict(X_scaled)

    df['label'] = labels

    return df

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

if __name__ == "__main__":

    df = pd.read_csv('../data/sample1000_with_lyrics.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~((df['instrumentalness'] < 0.9) & (df['lyrics'].isna()))].reset_index(drop = True)

    df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)
    #print(df[['track_name', 'clean_lyrics']].head())

    df = kmeans_model(df)
    pairs_df = create_pairs(df)
    model = finetune_bert(pairs_df, model_name = 'all-MiniLM-L6-v2', num_epochs = 3)
    model.save('../models/finetuned_bert')
