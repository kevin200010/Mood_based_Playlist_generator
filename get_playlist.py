#from spotify import *
import warnings
warnings.filterwarnings("ignore")
import spotipy
from spotipy.oauth2 import SpotifyOAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="6948f82efec347c28bc9fe000632e0b0",
                                               client_secret="633a43196bc84548b3031a9453307979",
                                               redirect_uri="http://google.com/",
                                               scope="user-read-recently-played"))


def get_playlist():
    playlist = sp.current_user_recently_played(limit=30, after=None, before=None)
    return playlist
    