# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:25:24 2020

@author: ichid
"""
######################## package import #################################

import os
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

os.chdir('D:/Documents/Travail perso/Exercices/Montagne_Meteo_France')

######################### data import (if not already loaded ###############################
try:
    if(table_all_dates_all_stations_batch_filled is not None):
        pass
except NameError:
    table_all_dates_all_stations_batch_filled = pd.read_excel('data/output/table_complete_morning_cleaned_batchs_filled.xlsx')









##############################################################################################################################
############################################## PREPROCESSING DATA FOR LEARNING  ##############################################
dataset = table_all_dates_all_stations_batch_filled.filter(items = ['Latitude_x', 
        'Altitude_x', '_vent_moy_10min_m/s', 'temperature', 'nebulosite', 
       'nebulosite_etage_inf', 'precipitations_24h', 'temperature_min_24h', 'temperature_max_24h', 'hauteur_neige', 
       'hauteur_neige_fraiche', 'batch', 'rayonnement_solaire'])


##### rescaling ######
scaler = MinMaxScaler(feature_range = (0,1))
dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset.drop(columns = ['batch'] )), 
                              columns = ['Latitude_x',  'Altitude_x', '_vent_moy_10min_m/s', 'temperature', 
                                         'nebulosite', 'nebulosite_etage_inf',  'precipitations_24h', 
                                         'temperature_min_24h', 'temperature_max_24h', 'hauteur_neige', 
                                         'hauteur_neige_fraiche', 'rayonnement_solaire'])
dataset_scaled['batch'] = dataset['batch'].reset_index(drop = True)
max_y = dataset['hauteur_neige'].max()


#### passage en array. Nombre de batchs (actuellement 335)
list_batches = dataset_scaled['batch'].unique()
nb_batches = len(list_batches)
nb_batches

#taille sequence (actuellement 122)
len_batch_sequence = dataset_scaled[dataset_scaled.batch == list_batches[0]].shape[0]
len_batch_sequence

#nombre param entree (actuellement 11)
nb_x_signals = dataset_scaled[dataset_scaled.batch == list_batches[0]].shape[1] - 2
nb_x_signals

#initialisation batchs vides
dataset_batches_x = np.empty([len(list_batches), 1])
dataset_batches_y = np.empty([len(list_batches), 1])

dataset_batches_x = np.empty(shape = (nb_batches, len_batch_sequence, nb_x_signals))
dataset_batches_y = np.empty(shape = (nb_batches, len_batch_sequence, 1))

for batch_num in range(0, len(list_batches)):
    data_batch_x  = dataset_scaled[dataset_scaled['batch'] == list_batches[batch_num] ].drop(columns = ['batch', 'hauteur_neige'])
    data_batch_array_x  = np.asanyarray(data_batch_x)
    dataset_batches_x[batch_num] = data_batch_array_x
    
    data_batch_y  = dataset_scaled[dataset_scaled['batch'] == list_batches[batch_num] ].filter(['hauteur_neige'])
    data_batch_array_y  = np.asanyarray(data_batch_y)
    dataset_batches_y[batch_num] = data_batch_array_y
    
#dimension des batchs x et y
print(dataset_batches_x.shape)
print(dataset_batches_y.shape)

#affichage des eventuels NA encore presentes. Attention, le moindre NA rendra l'entrainement impossible
print(np.argwhere(np.isnan(dataset_batches_x)))
print(np.argwhere(np.isnan(dataset_batches_y)))


#suppression donnees inutiles
del [data_batch_array_x,
     data_batch_array_y,
     data_batch_x,
     data_batch_y,
     table_all_dates_all_stations_batch_filled]


#train / test sets
train_batch_select = random.sample(range(0,nb_batches), int(0.9*nb_batches)-1)
#train batch select est la liste de rangs selectionnes aleatoirement pour le train.
#Donc partie entiere de 0,9*nombre de batches. Tous ceux non selectionnes seront pour les tests

#ecriture de cette selection. ATTENTION : dans ce cas enregistrer les modeles par la suite associes a cette selection aleatoire
with open("models/train_batch_select.txt", "wb") as fp:
    pickle.dump(train_batch_select, fp)
    
with open('models/dataset_batches_x.pkl', 'wb') as fx:
    pickle.dump(dataset_batches_x, fx)
    
with open('models/dataset_batches_y.pkl', 'wb') as fy:
    pickle.dump(dataset_batches_y, fy)
    
with open("models/list_batches.txt", "wb") as fp:
    pickle.dump(list_batches, fp) 
    
list_max_y = []
list_max_y.append(max_y)
with open("models/list_max_y.txt", "wb") as fm:
    pickle.dump(list_max_y, fm) 


#ayant maintenant le 'train_batch_select', on peut creer les tables de batchs train_x, test_x, train_y, test_y
train_x = dataset_batches_x[train_batch_select]
test_x = np.delete(dataset_batches_x, train_batch_select, axis = 0)
train_y = dataset_batches_y[train_batch_select]
test_y = np.delete(dataset_batches_y, train_batch_select, axis = 0)

test_data = (test_x, test_y)








####################################################################################################################
############################################## LEARNING  ###########################################################

################## DEEP LEARNING TEMPORAL SERIES - LSTM 20 units model  ##################
model_lstm_20 = tf.keras.Sequential()

model_lstm_20.add(tf.keras.layers.LSTM(
          units = 20,
          batch_input_shape = (len(train_batch_select), len_batch_sequence, nb_x_signals),
          stateful = False,
          activation = "tanh",
          use_bias = True,
          kernel_initializer = "glorot_uniform",
          recurrent_initializer = "orthogonal",
          bias_initializer = "zeros",
          return_sequences = True))

model_lstm_20.add(tf.keras.layers.Dense(1))

model_lstm_20.compile(optimizer="adam", loss="mean_squared_error")

model_lstm_20.summary()

model_lstm_20.fit(x = train_x,
                  y = train_y,
                  batch_size = len(train_batch_select),
                  epochs = 1000)

model_lstm_20.save('models/model_lstm_20')



################# DEEP LEARNING TEMPORAL SERIES - GRU 20 units model  ##################
model_gru_20 = tf.keras.Sequential()

model_gru_20.add(tf.keras.layers.GRU(
          units = 20,
          batch_input_shape = (nb_batches, len_batch_sequence, nb_x_signals),
          stateful = False,
          activation = "tanh",
          use_bias = True,
          kernel_initializer = "glorot_uniform",
          recurrent_initializer = "orthogonal",
          bias_initializer = "zeros",
          return_sequences = True))

model_gru_20.add(tf.keras.layers.Dense(1))

model_gru_20.compile(optimizer="adam", loss="mean_squared_error")

model_gru_20.summary()

model_gru_20.fit(x = train_x,
          y = train_y,
          batch_size = len(train_batch_select),
          epochs = 1000)

model_gru_20.save('models/model_gru_20')



################# DEEP LEARNING TEMPORAL SERIES - simple RNN 20 units model  ##################
model_simplernn_20 = tf.keras.Sequential()

model_simplernn_20.add(tf.keras.layers.SimpleRNN(
          units = 20,
          batch_input_shape = (nb_batches, len_batch_sequence, nb_x_signals),
          stateful = False,
          activation = "tanh",
          use_bias = True,
          kernel_initializer = "glorot_uniform",
          recurrent_initializer = "orthogonal",
          bias_initializer = "zeros",
          return_sequences = True))

model_simplernn_20.add(tf.keras.layers.Dense(1))

model_simplernn_20.compile(optimizer="adam", loss="mean_squared_error")

model_simplernn_20.summary()

model_simplernn_20.fit(x = train_x,
          y = train_y,
          batch_size = len(train_batch_select),
          epochs = 1000)

model_simplernn_20.save('models/model_simplernn_20')

