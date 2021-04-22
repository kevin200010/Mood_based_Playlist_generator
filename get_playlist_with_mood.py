from get_playlist import *
# from get_mood import *
from get_song_attribute import *
import pandas as pd

#mood = get_mood()

#importing playlist and conveting to list
# playlist = get_playlist()
# print(playlist)


# print(playlist)
# song_data = {'song_id' : playlist.item}


# cols = ['name' , 'id' , 'mood']
# song_list = pd.DataFrame(columns = cols)

# for  item in playlist['items']:
#     track = item['track']
#     print(track['artists'][0]['name'], " – ", track['name'],"-" ,track['id'])
#     #print(get_song_feature(track['id']) , "\n")
#     #song_list = song_list.append({ 'name' : track['name'] , 'id' : track['id'] , 'mood': 'notDefined'} , ignore_index =True)


# #print(song_list)



###############################################################################################################
from test import *
from spotify import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#for sequeantial model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_json


#importing tf and disable the v2 behavior and eager mode
 

#for validation of model
from sklearn.model_selection import cross_val_score , KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix , accuracy_score


#importing tf and disable the v2 behavior and eager mode
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()

# #load Dataframe and prepare
# df = pd.read_csv("song_data_moods.csv")

# # preparing data for model
# col_feature = df.columns[6 : -3]
# X = MinMaxScaler().fit_transform(df[col_feature])
# X2 = np.array(df[col_feature])
# Y = df['mood']

# #Encode data
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_y = encoder.transform(Y)


# # Convert to  dummy (Not necessary in my case)
# dummy_y = np_utils.to_categorical(encoded_y)
# X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
# target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)


# # model creation function
# def base_model():
#     #Create the model
#     model = Sequential()
#     #Add 1 layer with 8 nodes,input of 4 dim      with relu function
#     model.add(Dense(8,input_dim=10,activation='relu'))
#     #Add 1 layer with output 3 and softmax function
#     model.add(Dense(4,activation='softmax'))
#     #Compile the model using sigmoid loss function an adam optim
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     return model

# # estimation classifier with base model
# estimator = KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0)

# #Evaluate the model using KFold cross validation
# kfold = KFold(n_splits=10,shuffle=True)
# results = cross_val_score(estimator,X,encoded_y,cv=kfold)
# #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

# estimator.fit(X_train,Y_train)
# y_preds = estimator.predict(X_test)


# #Join the model and the scaler in a Pipeline
# pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0))])
# #Fit the Pipeline
# pip.fit(X2,encoded_y)


# #creating song_list for result
# # results = sp.current_user_recently_played(limit=30, after=None, before=None)
# # # print(results['items'])
# cols = ['name' , 'id' , 'mood']
# song_list = pd.DataFrame(columns = cols)


# for idx, item in enumerate(playlist['items']):
#     track = item['track']
#     # print(idx, track['artists'][0]['name'], " – ", track['name'],"-" ,track['id'])
#     #print(get_song_feature(track['id']) , "\n")
#     ids = track['id']
#     preds = get_song_feature(ids)
#     preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T

#     #Predict the features of the song
#     results = pip.predict(preds_features)

#     mood = np.array(target['mood'][target['encode']==int(results)])
#     name_song = preds[0][0]
#     artist = preds[0][2]
#     song_list = song_list.append({ 'name' : track['name'] , 'id' : track['id'] , 'mood': mood[0]} , ignore_index =True)
#     # print("{0} by {1} is a {2} song".format(name_song,artist,mood[0].upper()))

# print(song_list)

# #####################################################################################################################33
#####################################################################################################################   

def get_playlist_with_mood():
    playlist = get_playlist()
    # print(playlist)
    warnings.filterwarnings("ignore")
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    
    # song_data = {'song_id' : playlist.item}
    # cols = ['name' , 'id' , 'mood']
    # song_list = pd.DataFrame(columns = cols)

    # for  item in playlist['items']:
    #     track = item['track']
    #     print(track['artists'][0]['name'], " – ", track['name'],"-" ,track['id'])
        #print(get_song_feature(track['id']) , "\n")
        #song_list = song_list.append({ 'name' : track['name'] , 'id' : track['id'] , 'mood': 'notDefined'} , ignore_index =True)
    #print(song_list)
    ###############################################################################################################
    # from test import *
    # from spotify import *
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # #for sequeantial model
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.wrappers.scikit_learn import KerasClassifier
    # from keras.utils import np_utils
    # from keras.models import model_from_json
    # #importing tf and disable the v2 behavior and eager mode
 

    # #for validation of model
    # from sklearn.model_selection import cross_val_score , KFold , train_test_split
    # from sklearn.preprocessing import LabelEncoder , MinMaxScaler
    # from sklearn.pipeline import Pipeline
    # from sklearn.metrics import confusion_matrix , accuracy_score


    # #importing tf and disable the v2 behavior and eager mode
    # import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_v2_behavior()

    #load Dataframe and prepare
    df = pd.read_csv("song_data_moods.csv")

    # preparing data for model
    col_feature = df.columns[6 : -3]
    X = MinMaxScaler().fit_transform(df[col_feature])
    X2 = np.array(df[col_feature])
    Y = df['mood']

    #Encode data
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)


    # Convert to  dummy (Not necessary in my case)
    dummy_y = np_utils.to_categorical(encoded_y)
    X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)


    # model creation function
    def base_model():
        #Create the model
        model = Sequential()
        #Add 1 layer with 8 nodes,input of 4 dim      with relu function
        model.add(Dense(8,input_dim=10,activation='relu'))
        #Add 1 layer with output 3 and softmax function
        model.add(Dense(4,activation='softmax'))
        #Compile the model using sigmoid loss function an adam optim
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    # estimation classifier with base model
    estimator = KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0)

    #Evaluate the model using KFold cross validation
    kfold = KFold(n_splits=10,shuffle=True)
    results = cross_val_score(estimator,X,encoded_y,cv=kfold)
    #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))


    estimator.fit(X_train,Y_train)
    y_preds = estimator.predict(X_test)


    #Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0))])
    #Fit the Pipeline
    pip.fit(X2,encoded_y)


    #creating song_list for result
    # results = sp.current_user_recently_played(limit=30, after=None, before=None)
    # # print(results['items'])
    cols = ['name' , 'id' , 'mood']
    song_list = pd.DataFrame(columns = cols)


    for idx, item in enumerate(playlist['items']):
        track = item['track']
        # print(idx, track['artists'][0]['name'], " – ", track['name'],"-" ,track['id'])
        #print(get_song_feature(track['id']) , "\n")
        ids = track['id']
        preds = get_song_feature(ids)
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        #Predict the features of the song
        results = pip.predict(preds_features)

        mood = np.array(target['mood'][target['encode']==int(results)])
        #name_song = preds[0][0]
        #artist = preds[0][2]
        song_list = song_list.append({ 'name' : track['name'] , 'id' : track['id'] , 'mood': mood[0]} , ignore_index =True)
        # print("{0} by {1} is a {2} song".format(name_song,artist,mood[0].upper()))

    # print(song_list)
    return song_list
