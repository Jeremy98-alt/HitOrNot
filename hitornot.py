#libraries used for machine learning  
import numpy as np
import pandas as pd

import sklearn
from sklearn.cluster import KMeans

from sklearn import datasets
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from matplotlib import pyplot as plt

#libraries used for managing the main OS functions
import os
import time
import sys
import math

#spotipy library, this is to use different functions and the access to the API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

#these libraries is used to manage the main format data
import json
from json.decoder import JSONDecodeError
import csv

def get_spotify_token():
    #get username from terminal, user ID is: 1168261438
    username = sys.argv[1]

    #set up a scope, see in https://developer.spotify.com/documentation/general/guides/scopes/
    scope = '' 

    token = util.prompt_for_user_token(username,
                                    scope, 
                                    client_id='b3e30f50066f47c683dc47c1ab15526c',
                                    client_secret='fefca33bfe9047e78ab42c2ad24fb4a3',
                                    redirect_uri='https://google.com')

    #type access: Authorization Code Flow
    if token:
        spotify = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)

    return spotify

#Z-Scaling used for normalized the ranges, First Part
def normalized_dt(x):
    normalized_dataset = x.copy()
    normalized_dataset[['Total Stream', 'MediumPositionsINTERN', 'PresenceIN','Markets','Tempo']] = (x[['Total Stream', 'MediumPositionsINTERN', 'PresenceIN','Markets','Tempo']] - x[['Total Stream', 'MediumPositionsINTERN', 'PresenceIN','Markets','Tempo']].mean())/x[['Total Stream', 'MediumPositionsINTERN', 'PresenceIN','Markets','Tempo']].std()
    
    print(normalized_dataset.head())
    print(normalized_dataset.describe()) #get information about dataset, count, mean, std, min, quantil, max..

    return normalized_dataset

def get_information_newsong(spotify, normalized_dataset):
    search_uri_song = "none"
    search_uri_artist = "none"
    search_numbermarkets = 0

    aa = spotify.search(search)
    items = aa['tracks']['items']
    for a in items:
        if a['artists'][0]['name'] == search_artist:
            if a['name'] == search_song:
                print(" \nSong found, i'm catching the song and artist uri")
                search_uri_song = a['uri']
                search_uri_artist = a['artists'][0]['uri']
                search_numbermarkets = len(a['available_markets'])

    #get audio features and genre
    temp = {}
    if search_uri_artist and search_uri_song != "none":

        aa = spotify.artist(search_uri_artist)
        if aa['genres']:
            found_genre = aa['genres'][0]
        else:
            found_genre = "no genres"

        bb = spotify.audio_features(search_uri_song)
        for x in bb: 
            temp = {
                'Acousticness' : x['acousticness'],
                'Danceability' : x['danceability'],
                'Energy' : x['energy'],
                'Instrumentalness' : x['instrumentalness'],
                'Liveness' : x['liveness'],
                'Loudness' : ((x['loudness'] - dataset[['Loudness']].mean())/dataset[['Loudness']].std()).iloc[0],
                'Speechiness' : x['speechiness'],
                'Valence' : x['valence'],
                'Key' : x['key'],
                'Mode' : x['mode'],
                'Duration_ms' : ((x['duration_ms'] - dataset[['Duration_ms']].mean())/dataset[['Duration_ms']].std()).iloc[0],
                'Tempo' : ((x['tempo'] - dataset[['Tempo']].mean())/dataset[['Tempo']].std()).iloc[0]
            }
            
    search_numbermarkets = ((search_numbermarkets - dataset[['Markets']].mean())/dataset[['Markets']].std()).iloc[0]

    row_to_add = pd.DataFrame( { 
                        'Artist': [search_artist],
                        'Song': [search_song],
                        'Total Stream': 1,
                        'MediumPositionsINTERN': 1,
                        'PresenceIN': 1, 
                        'Markets':[search_numbermarkets],
                        'Acousticness':[temp['Acousticness']],
                        'Danceability': [temp['Danceability']],
                        'Energy': [temp['Energy']],
                        'Instrumentalness': [temp['Instrumentalness']],
                        'Liveness': [temp['Liveness']],
                        'Loudness': [temp['Loudness']],
                        'Speechiness': [temp['Speechiness']],
                        'Valence': [temp['Valence']],
                        'Key': [temp['Key']],
                        'Mode': [temp['Mode']],
                        'Duration_ms': [temp['Duration_ms']],
                        'Tempo': [temp['Tempo']],
                        'Genre': [found_genre] 
                        } )

    normalized_dataset = normalized_dataset.append(row_to_add, ignore_index=True, sort=False)

    return row_to_add

def MAE(y_true, y_pred):
    return (y_true-y_pred).abs().mean()

def kNNAlgorithm(normalized_dataset, row_to_add):
    np.random.seed(123)

    normalized_dataset = normalized_dataset.drop(len(normalized_dataset)-1)

    X = normalized_dataset.drop(['Song','Artist','Genre','Duration_ms','Total Stream'], axis = 1)
    Y = normalized_dataset[['Total Stream']]

    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    max_score = 0
    k_max = 1
    for x in range(2,9):
        neigh = KNeighborsRegressor(n_neighbors=x)
        neigh = neigh.fit(x_train,y_train)
        if max_score < neigh.score(x_test, y_test):
            max_score = neigh.score(x_test, y_test)
            k_max = x

    print(f"Best score is: {max_score} with k = {k_max} close data")
    neigh = KNeighborsRegressor(n_neighbors=k_max)
    neigh = neigh.fit(x_train,y_train)

    y_train_preds = neigh.predict(x_train)
    y_test_preds = neigh.predict(x_test)

    #only to display the dataframe.
    y_dtest_preds = pd.DataFrame({'TotalStream_predicts': y_test_preds[:, 0]})
    y_dtest_preds.tail(10)

    x_test.tail(10) #show variables indipendent

    new_x_test = row_to_add.drop(['Song','Artist','Genre','Duration_ms','Total Stream'], axis = 1)
    new_y_test_preds = neigh.predict(new_x_test)

    #only to display the dataframe, normalize the value.
    y_dtest_preds = pd.DataFrame({'TotalStream_predicts': math.ceil(new_y_test_preds[: ,0] * dataset[['Total Stream']].std()+dataset[['Total Stream']].mean()) }, index=[0])

    hypothetical_total_stream = int(y_dtest_preds.values[0])
    print(f"Total Stream: {hypothetical_total_stream}")

    print("MAE on the training set: {}".format(MAE(y_train,y_train_preds)))
    print("MAE on the test set: {}".format(MAE(y_test,y_test_preds)))

    return hypothetical_total_stream

if __name__ == '__main__':
    spotify = get_spotify_token()

    #load the dataset + regression linear
    dataset = pd.read_csv('dataset_for_proj.csv')
    dataset = dataset.drop(['MediumPositionsALL'], axis=1)

    print(dataset.info())  #get info about, dataset
    print(dataset.head()) #get the first five tuples
    print(dataset.describe()) #get information about dataset, count, mean, std, min, quantil, max..

    normalized_dataset = normalized_dt(dataset)

    #Search a song on spotify, so get information about this song
    search = input("This is a search bar put more information: ")

    search_song = input("Insert Song Name: ")
    search_artist = input("Insert Artist Name: ")

    check_artist = (search_artist == normalized_dataset.Artist).any()
    check_song = (search_song == normalized_dataset.Song).any()
    if check_song & check_artist:
        print("IS A HIT")
    else: 
        row_to_add = get_information_newsong(spotify, normalized_dataset)

        normalized_dataset[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness','Valence','Tempo']] /= 1000
        hypothetical_total_stream = kNNAlgorithm(normalized_dataset, row_to_add)

        print("\nThis is an Hit or Not?: ")

        if hypothetical_total_stream > 0 and hypothetical_total_stream < int(dataset[['Total Stream']].mean()/2):
            print("\nNOT AN HIT")
            print(f"Total Stream: {hypothetical_total_stream}")
        elif hypothetical_total_stream > int(dataset[['Total Stream']].mean()/2) and hypothetical_total_stream < int(dataset[['Total Stream']].mean()):
            print("\nLOCAL HIT")
            print(f"Total Stream: {hypothetical_total_stream}")
        else:
            print("\nMAYBE GLOBAL HIT!")
            print(f"Total Stream: {hypothetical_total_stream}")
