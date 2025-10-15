import lyricsgenius
import os
from dotenv import load_dotenv
import pandas as pd
import time
from tqdm import tqdm

def get_lyrics(access_token, df):

    lyrics_list = []

    genius = lyricsgenius.Genius(access_token, timeout=10)
    for i, row in tqdm(df.iterrows(), desc = 'Fetching lyrics', total = len(df)):

        try:
            song = genius.search_song(row['track_name'], row['track_artist'])
            if song:
                lyrics_list.append(song.lyrics)
            else:
                lyrics_list.append(None)
        except Exception as e:
            print(f"Error for {row['track_name']}: {e}")
            lyrics_list.append(None)
        time.sleep(1)
    
    return lyrics_list

if __name__ == "__main__":

    load_dotenv()
    access_token = os.getenv('GENIUS_ACCESS_TOKEN')
    df = pd.read_csv('../data/spotify_clean.csv')
    lyrics_list = get_lyrics(access_token, df)
    df['lyrics'] = lyrics_list
    df.csv('../data/spotify_clean_with_lyrics.csv', index = False)

    print('Lyrics collection complete')

