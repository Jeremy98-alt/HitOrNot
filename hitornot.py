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

'''
# find the common elements, in two lists 
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return list(a_set & b_set)
    else: 
        return 0 

def result_stats_new_song(combine_info, normalized_dataset, temp, dataset):
    list_of_none_features = []
    old_len = len(normalized_dataset) - 1
    normalized_dataset = normalized_dataset[normalized_dataset.Labels == normalized_dataset.loc[len(normalized_dataset)-1].Labels]
    normalized_dataset = normalized_dataset.drop(old_len)

    for x in combine_info:
        X = x[0]
        Y = x[1]

        #Plot the first values and add the new value, we want to analyze 10 songs
        print(plt.scatter( normalized_dataset[[X]], normalized_dataset[[Y]], color ="green"  ))
        print(plt.scatter( temp[X], temp[Y], color ="red"  ))
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.grid()

        print(plt.show())

        #zoom to the cluster 
        for i in range(normalized_dataset.shape[1]):   
            group_x = normalized_dataset.loc[ (normalized_dataset[X] < ( temp[X] + 0.05 )) & (normalized_dataset[X] > ( temp[X] - 0.05 )) , X]
            group_y = normalized_dataset.loc[ (normalized_dataset[Y] < ( temp[Y] + 0.05 )) & (normalized_dataset[Y] > ( temp[Y] - 0.05 )) , Y]

        around_values = common_member(list(group_x.index), list(group_y.index)) 

        if around_values:
            row_data = normalized_dataset.loc[around_values, :].copy()
            row_data[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']] = (row_data[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']]*dataset[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']].std() ) + dataset[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']].mean()

            ax = row_data.plot.scatter(x=list(row_data[[X]]), y=list(row_data[[Y]]), alpha=0.5)
            for i, txt in enumerate(row_data.Song):
                ax.annotate(txt, (row_data[X].iat[i],row_data[Y].iat[i]))

            print( plt.scatter( temp[X], temp[Y], color ="red"  ))
            plt.annotate(search_song, (temp[X], temp[Y]) )
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.grid()

            print(plt.show())

            total_stream_newartist = row_data[['Total Stream']].sum() / len(around_values)
            print(total_stream_newartist)

            MPI_newartist = row_data[['MediumPositionsINTERN']].sum() / len(around_values)
            print(MPI_newartist)

            MPA_newartist = row_data[['MediumPositionsALL']].sum() / len(around_values)
            print(MPA_newartist)

            PresenceIN_newartist = math.ceil(row_data[['PresenceIN']].sum() / len(around_values))
            print("PresenceIN: ", PresenceIN_newartist)

            list_of_none_features.append([total_stream_newartist,MPI_newartist,MPA_newartist,PresenceIN_newartist])
        else:
            row_data[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']] = ( normalized_dataset[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']]*dataset[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']].std() ) + dataset[['Total Stream','MediumPositionsINTERN','MediumPositionsALL','PresenceIN']].mean()
            print(row_data)

            total_stream_newartist = row_data[['Total Stream']].mean()
            print(total_stream_newartist)

            MPI_newartist = row_data[['MediumPositionsINTERN']].mean()
            print(MPI_newartist)

            MPA_newartist = row_data[['MediumPositionsALL']].mean()
            print(MPA_newartist)

            PresenceIN_newartist = math.ceil(row_data[['PresenceIN']].mean())
            print("PresenceIN: ", PresenceIN_newartist)

            list_of_none_features.append([total_stream_newartist,MPI_newartist,MPA_newartist,PresenceIN_newartist])

    return list_of_none_features
'''

'''   
        list_for_analysis = ['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence']
        combine_info = list(combinations(list_for_analysis, 2))

        list_of_none_features = result_stats_new_song(combine_info, normalized_dataset, temp, dataset)

        total_stream_newartist = 0
        MPI_newartist = 0
        MPA_newartist = 0
        PresenceIN_newartist = 0
        for x in list_of_none_features:
            for i,single_feature in enumerate(x):  
                if i == 0:
                    total_stream_newartist += single_feature
                elif i == 1:
                    MPI_newartist += single_feature
                elif i == 2:
                    MPA_newartist += single_feature
                else:
                    PresenceIN_newartist += single_feature
        
        total_stream_newartist = math.ceil( abs(total_stream_newartist)/ 2 )
        MPI_newartist = math.ceil(abs(MPI_newartist)/len(list_of_none_features))
        MPA_newartist = float( round(abs(MPA_newartist)/len(list_of_none_features),2) )
        PresenceIN_newartist = math.ceil(PresenceIN_newartist/len(list_of_none_features))  #no ceil?

        print("\nThis is an Hit or Not?: ")
        if total_stream_newartist > 1000 and total_stream_newartist < 500000:
            print("\nNOT AN HIT")
            print(f"Total Stream: {total_stream_newartist}, MPI: {MPI_newartist}, MPA: {MPA_newartist}, PresenceIN: {PresenceIN_newartist}")
        elif total_stream_newartist > 500000 and total_stream_newartist < 2000000:
            print("\nLOCAL HIT")
            print(f"Total Stream: {total_stream_newartist}, MPI: {MPI_newartist}, MPA: {MPA_newartist}, PresenceIN: {PresenceIN_newartist}")
        else:
            print("\nMAYBE GLOBAL HIT!")
            print(f"Total Stream: {total_stream_newartist}, MPI: {MPI_newartist}, MPA: {MPA_newartist}, PresenceIN: {PresenceIN_newartist}")
'''



'''
X = normalized_dataset[normalized_dataset.columns[7:19]]
variable_names = normalized_dataset[['Song']]

clustering = KMeans(n_clusters=4)
clustering = clustering.fit(X)

centroids = np.array(clustering.cluster_centers_)


plt.subplot(1,2,1)
plt.scatter(X[['Loudness']], X[['Liveness']], c = 'green', s=50)
plt.title('Ground Truth Classification')
plt.xlabel('Loudness')
plt.ylabel('Liveness')

plt.subplot(1,2,2)
plt.scatter(X[['Loudness']], X[['Liveness']], c=clustering.labels_.astype(float), s=50)
plt.scatter(centroids[: ,0], centroids[: ,1], marker = "x", color='r')
plt.title('K-Means Classification')
plt.xlabel('Loudness')
plt.ylabel('Liveness')

plt.show()'''

'''def silhoutte(X, data_reduced):
      for i, k in enumerate([2, 3, 4]):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        
        # Run the Kmeans algorithm
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        # Get silhouette samples
        silhouette_vals = silhouette_samples(X, labels)

        # Silhouette plot
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals)
        ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax1.set_yticks([])
        ax1.set_xlim([-0.1, 1])
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster labels')
        ax1.set_title('Silhouette plot for the various clusters', y=1.02)
        
        # Scatter plot of data colored with labels
        ax2.scatter(data_reduced[0], data_reduced[1], c=labels)
        ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
        ax2.set_xlim([-6, 6])
        ax2.set_ylim([-2, 2])
        ax2.set_xlabel('Eruption time in mins')
        ax2.set_ylabel('Waiting time to next eruption')
        ax2.set_title('Visualization of clustered data', y=1.02)
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.suptitle(f'Silhouette analysis using k = {k}',
                    fontsize=16, fontweight='semibold', y=1.05)
        
        plt.show()
'''

'''
        #KMeans Algorithm
        X = normalized_dataset[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence']]

        clustering = KMeans(n_clusters=2)
        labels = clustering.fit_predict(X)

        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(X)
        data_reduced = pd.DataFrame(data_reduced)

        ax = data_reduced.plot(kind='scatter', x=0, y=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('K-Means Classification (with PCA)')

        unique = list(set(labels))
        colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
        for i, u in enumerate(unique):
            xi = [data_reduced[0][j] for j  in range(len(data_reduced)) if labels[j] == u]
            yi = [data_reduced[1][j] for j  in range(len(data_reduced)) if labels[j] == u]
            plt.scatter(xi, yi, c=[colors[i]], label=str(u))
        
        normalized_dataset['Labels'] = labels
        for i, txt in enumerate(normalized_dataset.Song):
            if (normalized_dataset.loc[i,'Labels'] == normalized_dataset.loc[len(normalized_dataset)-1,'Labels']):

                if  ( (data_reduced[0].iat[i] < ( data_reduced[0].iat[len(normalized_dataset)-1] + 0.1 ) ) and (data_reduced[0].iat[i] > ( data_reduced[0].iat[len(normalized_dataset)-1] - 0.1 )) ):

                    if  ( (data_reduced[1].iat[i] < ( data_reduced[1].iat[len(normalized_dataset)-1] + 0.1 ) ) and (data_reduced[1].iat[i] > ( data_reduced[1].iat[len(normalized_dataset)-1] - 0.1 )) ):
                        ax.annotate(txt, (data_reduced[0].iat[i],data_reduced[1].iat[i]))

            if i == len(normalized_dataset) - 1:
                ax.annotate(txt, (data_reduced[0].iat[i],data_reduced[1].iat[i]))
                plt.scatter(data_reduced[0].iat[i],data_reduced[1].iat[i], c="yellow")  #only to display in the plot the new song

        plt.legend()

        plt.show()

        #when we convert the n-dimensional space into two dimensional space. We lose information about variance, in this mode we see the maintened information ... 97%
        print(pca.explained_variance_ratio_)

        #see the good cluster's number
        sse = []                                   #TO SEE THE GOOD NUMBER OF CLUSTERS
        list_k = list(range(1, 10))
        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(X)
            sse.append(km.inertia_)

        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Sum of squared distance')
        plt.show()

        #second type to Evaluation Method
        silhoutte(X, data_reduced)
'''

'''
        print("Inverse Z-Score")
        for idx, x in enumerate(y_test_preds):
            x = int((x*dataset[['Total Stream']].std()) + dataset[['Total Stream']].mean())
            y_test_preds[idx] = x
        
        print(y_test_preds)
'''