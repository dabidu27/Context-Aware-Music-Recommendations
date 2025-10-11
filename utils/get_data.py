import pandas as pd 
import kagglehub

#GET DATA
path = kagglehub.dataset_download("joebeachcapital/30000-spotify-songs")

print("Path to dataset files:", path)

sp_data = pd.read_csv('../data/spotify_songs.csv')

print(sp_data.columns)
print(sp_data.head(3))

#DATA CLEANING

print(sp_data.isna().sum())
sp_data = sp_data.dropna(subset=['track_name', 'track_artist', 'track_album_name'])

print(sp_data.isna().sum())
print('\n')
print(sp_data.dtypes)

print(sp_data.duplicated().sum())

numeric_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'track_popularity']
sp_data[numeric_cols] = sp_data[numeric_cols].apply(pd.to_numeric, errors = 'coerce')

print(sp_data.describe())

sp_data.to_csv('../data/spotify_clean.csv')


