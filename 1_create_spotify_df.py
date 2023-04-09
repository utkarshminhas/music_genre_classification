import os,json
from pprint import pprint
import spotipy,os
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd
import matplotlib.pyplot as plt


############################################################ pinging spotify api ############################################################

def ping_spotify_api():
    '''extracts details of the each song in a playlist from multiple playlists. puts the extracted details in a dataframe to be used later'''

    results = []
    exceptions = []
    for l in links:
        offset = [0,100]
        for off in offset:
            try:
                response = spotify.playlist_tracks(l['playlist_link'],limit=100, offset=off,)
            except Exception as e:
                l['exceptions'] = e
                exceptions.append(l)
                
                print("Exception in :")
                pprint(l)
                print("moving on to next iteration")
                print()
                continue # to next iteration

            if len(response['items']) > 0:
                print("num songs:",len(response['items']),"\t","pg:",str(off).zfill(3),"\t",l['playlist_name'],)
    
            for item in response['items']:
                artists = []
                for artist in item['track']['artists']:
                    artists.append(artist['name'])
                artists = sorted(artists)
                artists = ", ".join(artists)
            
                item_type = item['track']['type']
                song_name = item['track']['name']
                album_name = item['track']['album']['name']
                # release date is the same for the song as well as the album
                release_date = item['track']['album']['release_date']
                song_duration_ms = item['track']['duration_ms']
                popularity = item['track']['popularity']
                song_preview_url = item['track']['preview_url'] # so that we can verify the prediction by actually listening to it as well
                song_url = item['track']['external_urls']['spotify']
                popularity = item['track']['popularity']
            
                # release_date_precision = item['track']['album']['release_date_precision']
                genre = l['genre']

                results.append(
                    [
                        item_type,
                        song_name,
                        song_duration_ms,
                        artists,
                        album_name,
                        release_date,
                        song_preview_url,
                        song_url,
                        popularity,
                        genre,
                    ]
                )
    print("ALL exceptions")
    pprint(exceptions)
    print("excpetions over")

    return results


spotify_credentials = {
    "client_id":"c17b2cc765cd4ad9b5353574849a4496",
    "client_secret":"133d6fe314a6492ab94e04e967738fab"
}
client_id = spotify_credentials['client_id']
client_secret = spotify_credentials['client_secret']

client_credentials = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(client_credentials_manager= client_credentials)


with open('data/spotify_playlists.json') as f:
    links = json.load(f)

s = set()
for l in links:
    if l['genre'] != "":
        s.add(l['genre'])
#     if 'subgenres' in l.keys():
#         s.add(l['genre']+"'s sub: "+l['subgenres'])
print(len(s),"genres of songs found:",s)




results = ping_spotify_api()
df = pd.DataFrame(
    results,
    columns = [
        "item_type",
        "song_name",
        "song_duration_ms",
        "artists",
        "album_name",
        "release_date",
        "song_preview_url",
        "song_url",
        "popularity",
        "genre",
      ]
    )

print(df.shape)
df.to_csv('data/spotify_songs_details.csv',index=False)
