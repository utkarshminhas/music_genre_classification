import os, spotdl
from tqdm import tqdm
import pandas as pd

# df_path = 'data/top5_pre_genre_spotify_songs_details_cleaned.csv'
# save_dir = '/home/utkarsh/Desktop/music_genre_classification/data/sample_download/mp3_download'

# df_path = 'data/top75_per_genre_spotify_songs_details_cleaned.csv'
df_path = 'data/spotify_songs_details_cleaned.csv'
df = pd.read_csv(df_path)

# genres = df['genre'].value_counts().index.tolist()
# num_genres = len(genres)
# num_songs = df.shape[0]
# songs_per_genre = num_songs//num_genres # for floor division/rounding off
# save_dir = f'/home/utkarsh/Desktop/music_genre_classification/data/sample_download/mp3_download_top{songs_per_genre}'

save_dir = f'/home/utkarsh/Desktop/music_genre_classification/data/sample_download/mp3_download_all'


for index, row in tqdm(df.iterrows()):
    print(row['song_name'])
    print(row['genre'])
    save_path = os.path.join(save_dir,row['genre'])

    os.makedirs(save_path,exist_ok=True)
    print(row['song_url'])

    print('saving:',row['song_name'],"at:",save_path)
  
    command = f"spotdl {row['song_url']} --format mp3 --output {save_path}"
    os.system(command)