import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_lyrics(text):

    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text) #remove [verse], [chorus] etc 
    text = re.sub(r'\(.*?\)', ' ', text)  #remove 
    text = re.sub(r'[^a-z\s]', ' ', text)  #remove punctuation, numbers
    text = re.sub(r'\s+', ' ', text).strip() #collapse multiple spaces

    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


if __name__ == "__main__":

    df = pd.read_csv('../data/sample1000_with_lyrics.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~((df['instrumentalness'] < 0.9) & (df['lyrics'].isna()))].reset_index(drop = True)

    print(df.head())
    print(len(df))

    df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)
    print(df[['track_name', 'clean_lyrics']].head())

    print(df.isna().sum())



