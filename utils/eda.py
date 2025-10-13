import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/spotify_clean.csv')

print(df.shape)
print(df.describe)
print(df['playlist_genre'].value_counts())

#NUMERIC FEATURES

numeric_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'track_popularity']
#distributions

fig, axes = plt.subplots(3, 3, figsize = (12, 8))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):

    axes[idx].hist(df[col])
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout(pad = 2)
plt.savefig('../plots/audio_features_dist', dpi = 300)
#plt.show()

print(df['instrumentalness'].sort_values(ascending=False))

#boxplots

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'Boxplot of {col}')
    axes[idx].set_ylabel(col)

plt.tight_layout()

plt.savefig('../plots/audio_features_boxplots', dpi = 300)
#plt.show()

#CATEGORIAL FEATURES

categorical_cols = ['track_artist', 'playlist_name', 'playlist_genre', 'playlist_subgenre']

for col in categorical_cols:
    print(f'{col}: {df[col].nunique()} unique values')

#genre count

genre_counts = df['playlist_genre'].value_counts()
print(genre_counts)

plt.figure(figsize=(12, 8))
genre_counts.plot(kind = 'bar')

plt.title('Track Count By Genre')
plt.xlabel('Genre')
plt.ylabel('Number of tracks')
plt.xticks(rotation = 45)

plt.tight_layout()

plt.savefig('../plots/genres_barchart')
#plt.show()

#subgenre count

subgenres_counts = df['playlist_subgenre'].value_counts()
print(subgenres_counts)

plt.figure(figsize=(12, 8))
subgenres_counts.plot(kind = 'bar')

plt.title('Track Count By Subgenre')
plt.xlabel('Subgenre')
plt.ylabel('Number of tracks')
plt.xticks(rotation = 90)

plt.tight_layout()

plt.savefig('../plots/subgenres_barchart', dpi = 300)
#plt.show()


#CORRELATION ANALYSIS

correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot = True)

plt.suptitle('Correlation Matrix of Audio Features')
plt.tight_layout()

plt.savefig('../plots/correlation_matrix', dpi = 300)
#plt.show()

#strong correlation

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):

        if(abs(correlation_matrix.iloc[i, j]) > 0.5):
            print(f'Strong correlation: {correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]: .3f}')



