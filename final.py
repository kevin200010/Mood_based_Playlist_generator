from get_face_mood import *
from get_playlist_with_mood import *
import pandas as pd

#face_mood ='Angry' ,'Happy','Neutral','Sad','Surprise''Angry' ,'Happy','Neutral','Sad','Surprise'
mood = get_face_mood()
# print(mood)

#song_mood =  calm, energetic , happy ,sad 
song_list = get_playlist_with_mood()
print(song_list , "\n")

if mood == 'Sad':
    print("your mood is Sad , we prepare below playlist for you" , "\n")
    sad_song = song_list[song_list['mood'] == 'Sad']
    calm_song = song_list[song_list['mood'] == 'Calm']
    happy_song = song_list[song_list['mood'] == 'Happy']
    energetic_song = song_list[song_list['mood'] == 'Energetic']
    final_list = pd.concat([sad_song , calm_song , happy_song , energetic_song] , ignore_index = True)
    print(final_list)

elif mood == 'Happy':
    print("your mood is Happy , we prepare below playlist for you" , "\n")
    happy_song = song_list[song_list['mood'] == 'Happy']
    energetic_song = song_list[song_list['mood'] == 'Energetic']
    final_list = pd.concat([ happy_song , energetic_song] , ignore_index = True)
    print(final_list)

elif mood == 'Surprise':
    print("your are Surprised , we prepare below playlist for you" , "\n")
    calm_song = song_list[song_list['mood'] == 'Calm']
    energetic_song = song_list[song_list['mood'] == 'Energetic']
    final_list = pd.concat([calm_song , energetic_song] , ignore_index = True)
    print(final_list)

elif mood == 'Angry':
    print("why your are Angry , we prepare below playlist for you" , "\n")
    calm_song = song_list[song_list['mood'] == 'Calm']
    happy_song = song_list[song_list['mood'] == 'Happy']
    energetic_song = song_list[song_list['mood'] == 'Energetic']
    final_list = pd.concat([calm_song , happy_song , energetic_song] , ignore_index = True)
    print(final_list)

else:
    print("your mood is Neutral , we prepare below playlist for you" , "\n")
    energetic_song = song_list[song_list['mood'] == 'Energetic']
    final_list = pd.concat([energetic_song] , ignore_index = True)
    print(final_list)