import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

cid = '6948f82efec347c28bc9fe000632e0b0'
secret = '633a43196bc84548b3031a9453307979'

client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

urn = 'spotify:artist:3jOstUTkEu2JkjvRdBA5Gu'
artist = sp.artist(urn)
# print(artist)

user = sp.user('31lnxucgpqa5wyf7isneoou7hefu')
# print(user)

## fetching feature

def get_song_feature(ids):
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']    
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness','liveness','valence','loudness','speechiness','tempo','key','time_signature']

    return track , columns
# ids = '6HGoVbCUr63SgU3TjxEVj6'

# track , feature = get_song_feature(ids)
# print(track)

