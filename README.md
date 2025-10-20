 # Conditional Neural Network for Lyrics Embedding

This repository explores unsupervised learning (clustering) to discover latent audio groups, and a conditional neural generator that predicts lyrics embeddings from audio features + cluster labels.

The project has three main parts:

- Data preparation and exploratory analysis (in `utils/`)
- Unsupervised clustering of audio features to create contextual labels
- A conditional neural model that maps audio + cluster context to lyrics embeddings

---

## Table of contents

1. [Dataset](#1-dataset)
2. [Project layout](#2-project-layout)
3. [Clustering pipeline](#3-clustering-pipeline)
   3.1. [Preprocessing](#31-preprocessing)
   3.2. [Dimensionality reduction](#32-dimensionality-reduction)
   3.3. [Clustering algorithms tried](#33-clustering-algorithms-tried)
   3.4. [Validating clusters (silhouette score)](#34-validating-clusters-silhouette-score)
4. [Improving silhouette scores (practical tips)](#4-improving-silhouette-scores-practical-tips)
5. [Conditional neural generator](#5-conditional-neural-generator)
6. [How to reproduce experiments](#6-how-to-reproduce-experiments)
7. [Notes, limitations, and next steps](#7-notes-limitations-and-next-steps)
8. [Contributing](#8-contributing)
9. [License](#9-license)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Dataset

- The repository includes a cleaned Spotify dataset at `data/spotify_clean.csv` and a smaller sample with lyrics at `data/sample1000_with_lyrics.csv`.
- Relevant audio features used for clustering include: `danceability`, `energy`, `valence`, `mode`, `tempo`, `acousticness`, `speechiness`, `loudness`, `instrumentalness`.

## 2. Project layout

- `data/` - CSV files used for experiments
- `models/` - trained models and finetuned sentence-transformer
- `utils/` - scripts for EDA, cleaning lyrics, clustering, model training and orchestration
  - `model.py` - clustering pipeline, outlier removal, dataset and conditional generator, training loop
  - `ml.py` - clustering experiments and helpers (PCA, DBSCAN, kmeans loops)
  - `eda.py` - exploratory plots and correlation analysis
  - `clean_lyrics.py` - text cleaning utilities
  - `genius.py` - script to download lyrics from Genius API
  - `main.py` - example inference pipeline that loads the trained generator and produces predicted lyrics embeddings

## 3. Clustering pipeline

This section describes the pipeline used to produce cluster labels from audio features.

### 3.1. Preprocessing

- Feature selection: choose informative, low-redundancy features. The code commonly uses `['danceability','energy','valence','mode','tempo','speechiness','instrumentalness']`.
- Outlier removal: multivariate IQR-based filtering is applied in `model.py` (`remove_outliers`). This stabilizes clustering.
- Scaling: `StandardScaler` is used as a default. Alternatives (RobustScaler, MinMaxScaler, PowerTransformer) are recommended in experiments.

### 3.2. Dimensionality reduction

- PCA is used to reduce noise and redundancy (the repo experiments with 3 components and also finds components that retain ~90% variance).
- Consider non-linear embeddings (UMAP/t-SNE) for visualization or as a preprocessing step for clustering.

### 3.3. Clustering algorithms tried

- KMeans (primary in `model.py`, k=4 in the example pipeline)
- DBSCAN (density-based) — `utils/ml.py` contains k-distance plotting helpers to help select `eps`
- HDBSCAN (recommended where densities vary)
- AgglomerativeClustering / SpectralClustering / GaussianMixture (soft clusters)

### 3.4. Validating clusters (silhouette score)

- Global silhouette score (`sklearn.metrics.silhouette_score`) is used as a quantitative measure of cohesion vs separation.
- Per-sample silhouette (`silhouette_samples`) and silhouette plots are recommended for diagnostics to find boundary/noise points.

## 4. Improving silhouette scores (practical tips)

> Silhouette ranges from -1 to +1. Higher is better. Below are practical steps you can apply in this repo.

1. Preprocessing and scaling
   - Try `StandardScaler`, `RobustScaler`, `MinMaxScaler`, and `PowerTransformer`/Yeo-Johnson.
   - Revisit outlier thresholds: stricter or looser outlier removal can improve separability.

2. Feature selection & engineering
   - Drop or combine highly correlated features (see `utils/eda.py` correlation heatmaps).
   - Create derived features (tempo bins, loudness normalization, interactions like danceability*energy).

3. Dimensionality reduction
   - Grid-search number of PCA components and measure silhouette.
   - Try UMAP for preprocessing — it often improves clustering for non-linear structure.

4. Algorithm & hyperparameter tuning
   - Sweep `k` for KMeans (e.g., 2..12) and also try different algorithms like HDBSCAN for variable density.
   - Use k-distance plots to guide `eps` for DBSCAN (helpers exist in `utils/ml.py`).

5. Distance metric
   - Use cosine distance for textual/embedding-like data; Euclidean for standardized continuous features. `silhouette_score` supports precomputed distance matrices.

6. Remove or reassign low-scoring samples
   - Inspect per-sample silhouette. Low or negative values often indicate noise/boundary points; removing them can raise the global score.

7. Ensemble & stability
   - Consensus or ensemble clustering (multiple initializations/algorithms) and keeping stable assignments can improve robustness.

8. Balance cluster sizes
   - Extremely small clusters can drag down silhouette — consider merging tiny clusters or re-tuning `k`.

9. Visualization & diagnostics
   - Produce silhouette plots, PCA/UMAP scatter plots colored by cluster, and inspect cluster centers.

10. Cross-validate
   - Use subsampling/bootstrapping to verify that high silhouette generalizes.

## 5. Conditional neural generator

- The repo contains `ConditionalGenerator` (feedforward NN in `utils/model.py`) that maps audio features + cluster one-hot -> lyrics embedding (from a sentence-transformer).
- Training uses MSE loss and saves the best model to `models/best_model.pth`.

## 6. How to reproduce experiments

### 6.1. Set up environment

Create a Python virtual environment and install dependencies. Example (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages:

```powershell
pip install scikit-learn pandas numpy matplotlib seaborn torch sentence-transformers lyricsgenius tqdm python-dotenv nltk hdbscan umap-learn
```

### 6.2. Prepare data

- `utils/get_data.py` can (optionally) download and preprocess the dataset from Kaggle (requires access and `kagglehub`). The repo already includes `data/spotify_clean.csv`.
- To collect lyrics, set `GENIUS_ACCESS_TOKEN` in a `.env` file and run `utils/genius.py` (slow; rate-limited).

### 6.3. Run the pipeline

```powershell
python utils/clean_lyrics.py
python utils/model.py
python utils/main.py
```

Notes: `utils/model.py` performs clustering, creates embeddings (using the sentence-transformer in `models/finetuned_bert`) and trains the conditional generator.

## 7. Notes, limitations and next steps

- Silhouette is an internal clustering metric — verify clusters using domain checks and downstream metrics (recommendation performance).
- Add richer audio features (spectrogram-based features, MFCCs) and contextual metadata (user, playlist, time) for improved clusters.
- Try HDBSCAN + UMAP for better clusters on non-linear, variable-density data.
- Add notebooks that document experiment runs and visualization outputs.

## 8. Contributing

Contributions welcome: open a PR with focused changes (e.g., add silhouette analysis scripts, notebooks, tests).

## 9. License

This project is released under the LICENSE file in the repository root.

## 10. Acknowledgements

This project uses datasets and models from public sources including Kaggle and sentence-transformers.
