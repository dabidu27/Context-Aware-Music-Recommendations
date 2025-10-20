import pandas as pd
from model import kmeans_model, ConditionalGenerator
import torch
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def main():
    df = pd.read_csv('../data/spotify_clean.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df = df.sample(1000)
    df = kmeans_model(df)

    best_model = ConditionalGenerator(input_dim=7 + df['label'].nunique(), output_dim=384)
    best_model.load_state_dict(torch.load('../models/best_model.pth'))
    best_model.eval()


    audio_features = df[['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness']].values
    cluster_labels = df['label'].values.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    cluster_onehot = encoder.fit_transform(cluster_labels)

    combined_features = np.concatenate([audio_features, cluster_onehot], axis=1)
    combined_features = torch.FloatTensor(combined_features)

    with torch.no_grad():

        df['predicted_lyrics_embedding'] = best_model(combined_features).cpu().numpy().tolist()
    
    print(df.head())
    
if __name__ == '__main__':
    main()