from clean_lyrics import clean_lyrics
import pandas as pd
from sentence_transformers import InputExample, losses, SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

class AudioLyricsDataset(Dataset):

    def __init__(self, audio_features, lyrics_embedding, cluster_labels):

        audio_features = np.array(audio_features, dtype=np.float32)
        lyrics_embedding = np.stack(lyrics_embedding) 
        cluster_labels = np.array(cluster_labels).reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False)
        cluster_onehot = encoder.fit_transform(cluster_labels)

        combined_features = np.concatenate([audio_features, cluster_onehot], axis = 1)

        self.audio_features = torch.FloatTensor(combined_features)
        self.lyrics_embedding = torch.FloatTensor(lyrics_embedding)
    
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        return self.audio_features[idx], self.lyrics_embedding[idx]
    
class ConditionalGenerator(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims = [512, 256,  128]):

        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):

        return  self.network(x)

def train_model(model, df, epochs = 100, lr = 0.001, device = 'cpu', patience = 10):

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience=5, verbose = True)

    criterion = nn.MSELoss()
    cos_sim = nn.CosineSimilarity(dim = 1)

    history = {'train_loss': [], 'val_loss': [], 'train_cosine': [], 'val_cosine': []}

    best_val_loss = float('inf')
    patience_counter = 0

    audio_features = df[['danceability', 'energy', 'valence', 'mode', 'tempo', 'speechiness', 'instrumentalness']].values
    lyrics_embedding = df['lyrics_embedding'].values
    cluster_labels = df['label'].values

    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_dataset = AudioLyricsDataset(audio_features=audio_features[train_idx], lyrics_embedding=lyrics_embedding[train_idx], cluster_labels=cluster_labels[train_idx])
    val_dataset = AudioLyricsDataset(audio_features=audio_features[val_idx], lyrics_embedding=lyrics_embedding[val_idx], cluster_labels=cluster_labels[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        train_cosine = 0

        for audio, lyrics in train_loader:

            audio, lyrics = audio.to(device), lyrics.to(device)

            optimizer.zero_grad()

            pred_lyrics = model(audio)
            loss = criterion(pred_lyrics, lyrics)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            train_cosine = train_cosine + cos_sim(pred_lyrics, lyrics).mean().item()
    
        train_loss = train_loss/len(train_loader)
        train_cosine = train_cosine/len(train_loader)

        model.eval()
        
        val_loss = 0
        val_cosine = 0

        with torch.no_grad():

            for audio, lyrics in val_loader:

                audio, lyrics = audio.to(device), lyrics.to(device)
                pred_lyrics = model(audio)
                loss = criterion(pred_lyrics, lyrics)
                
                val_loss = val_loss + loss.item()
                val_cosine = cos_sim(pred_lyrics, lyrics).mean().item()
        
        val_loss = val_loss/len(val_loader)
        val_cosine = val_cosine/len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cosine'].append(train_cosine)
        history['val_cosine'].append(val_cosine)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '../models/best_model.pth')
        
        else:
            patience_counter = patience_counter  + 1

        if patience_counter > patience:

            print(f"Early stopping at epoch {epoch+1}")
            break
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Cosine: {train_cosine:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Cosine: {val_cosine:.4f}")
    

    model.load_state_dict(torch.load('../models/best_model.pth'))

    return model, history

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
    print(df.dtypes)

    """
    s = 0
    for i in range(10):
        s1 = df[df['label'] == 1].sample(1).iloc[0]['lyrics_embedding']
        s2 = df[df['label'] == 1].sample(1).iloc[0]['lyrics_embedding']
        similarity = cosine_similarity([s1], [s2])[0][0]
        print(similarity)

    """