from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import time
import pandas as pd 

load_dotenv()
spotify_id = os.getenv('SPOTIFY_CLIENT_ID')
spotify_secret = os.getenv('SPOTIFY_CLIENT_SECRET')


auth_manager = SpotifyClientCredentials(client_id=spotify_id, client_secret=spotify_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)


available_genres = sp.recommendation_genre_seeds()
print(available_genres)

def get_recommendations_for_genre(genre, n_songs):

    n_loops = n_songs//100

    seen_ids = set()
    tracks = []

    for _ in tqdm(range(n_loops), desc = f"{genre}"):

        try:

            recommendations = sp.recommendations(seed_genres = [genre], limit = 100)
            for track in recommendations['tracks']:

                track_id = track['id']
                if track_id and track_id not in seen_ids:

                    track_data = {

                        'track_id': track_id,
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'artist_id': track['artists'][0]['id'],
                        'popularity': track['popularity']

                    }

                    tracks.append(track_data)
                    seen_ids.add(track_id)
            
            time.sleep(0.5)

        except Exception as e:

            print('Error', e)
            time.sleep(1)
            continue

    df = pd.DataFrame(tracks)

    return df
