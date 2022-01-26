# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:23:54 2021

@author: ichid
"""


#import packages 
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tensorflow as tf

os.chdir('D:/Documents/Travail perso/Exercices/Montagne_Meteo_France')


####################################################################################################################
############################################## PREDICT, PLOT, COMPARE THE MODELS  ##################################

### premiere etape : si vous avez ouvert directement ce script sans avoir execute le 'learning.py', alors il faudra charger les modeles et donnees suivantes

#importing des modeles si ils n'y sont pas deja  
#lstm
try:
    if(model_lstm_20 is not None):
        pass
except NameError:
    model_lstm_20 = tf.keras.models.load_model('models/model_lstm_20')

#gru
try:
    if(model_gru_20 is not None):
        pass
except NameError:
    model_gru_20 = tf.keras.models.load_model('models/model_gru_20')

#simple RNN
try:
    if(model_simplernn_20 is not None):
        pass
except NameError:
    model_simplernn_20 = tf.keras.models.load_model('models/model_simplernn_20')
    
#CNN 1D
try:
    if(model_cnn is not None):
        pass
except NameError:
    model_cnn = tf.keras.models.load_model('models/model_cnn')
    
    
    


#x batches import (ceux utilises pour l'entrainement)
try:
    if(dataset_batches_x is not None):
        pass
except NameError:
    with open("models/dataset_batches_x.pkl", "rb") as fx:   # Unpickling
        dataset_batches_x = pickle.load(fx)

#y batches import (ceux utilises pour l'entrainement)
try:
    if(dataset_batches_y is not None):
        pass
except NameError:
    with open("models/dataset_batches_y.pkl", "rb") as fy:   # Unpickling
        dataset_batches_y = pickle.load(fy)

#train batch select (because it is a random selection)
try:
    if(train_batch_select is not None):
        pass
except NameError:
    with open("models/train_batch_select.txt", "rb") as fp:   # Unpickling
        train_batch_select = pickle.load(fp)

#list batches names
try:
    if(list_batches is not None):
        pass
except NameError:
    with open("models/list_batches.txt", "rb") as fb:   # Unpickling
        list_batches = pickle.load(fb)
        
#max y (= heuteur de neige max)
try:
    if(max_y is not None):
        pass
except NameError:
    with open("models/list_max_y.txt", "rb") as fm:   # Unpickling
        list_max_y = pickle.load(fm)
        max_y = list_max_y[0] 


#table des batches (avant standardisation)
try:
    if(table_all_dates_all_stations_batch_filled is not None):
        pass
except NameError:
    table_all_dates_all_stations_batch_filled = pd.read_excel('data/output/table_complete_morning_cleaned_batchs_filled.xlsx')



### fin de la reouverture des donnees et modeles necessaires !

#recalcul du nb de batchs et de leur longueur
nb_batches = len(list_batches)
len_batch_sequence = dataset_batches_x.shape[1]

#reexecution des fonctions de constitution des train_x, test_x, train_y, test_y
train_x = dataset_batches_x[train_batch_select]
test_x = np.delete(dataset_batches_x, train_batch_select, axis = 0)
train_y = dataset_batches_y[train_batch_select]
test_y = np.delete(dataset_batches_y, train_batch_select, axis = 0)





################ PREDICTION AVEC MODELE "NAIF" ####### -pred, plot and save-
#on prend la table avant standardisation, sur laquelle on applique un modele "naif". 
#Il ny a pas besoin de standardiser, puisqu'on va faire un modele simple sans apprentissage statistique

#le modele est simple :
# au jour 0 (=15 decembre), on cale y_pred = y^
#ensuite, on met y(t) = y(t-1) + hauteur_neige_fraiche(t-1) + {fonte_neige = -0,015 si decembre, janvier fevrier,
#                                                              fonte_neige = -0,03 si mars, avril}

#on a donc seulement la hausse de hauteur de neige qui se fait par une addition de neige fraiche,
#et la fonte qui se fait par une valeur constante de 0,015 ou 0,03 (determinée approximativement en faisant plusieurs essais)

error_list_naive_model = []

for batch in table_all_dates_all_stations_batch_filled.batch.unique():

    data_batch = table_all_dates_all_stations_batch_filled[table_all_dates_all_stations_batch_filled.batch == batch]
    data_batch = data_batch.reset_index(drop=True)
    data_batch['ht_neige_naive_model'] = np.nan
    data_batch.loc[0, ['ht_neige_naive_model']] = data_batch.loc[0, ['hauteur_neige']].item()

    sum_error_model = 0 
    
    for i in range(1, data_batch.shape[0]):
        if ((data_batch.loc[i, ['date']].item().month > 1) & (data_batch.loc[i, ['date']].item().month < 5)):
            melt = 0.03
        else:
            melt = 0.015
    
        ht_neige_naive_model = data_batch.loc[i-1, ['ht_neige_naive_model']].item() + data_batch.loc[i, ['hauteur_neige_fraiche']].item() - melt
        data_batch.loc[i, ['ht_neige_naive_model']] = ht_neige_naive_model
        
        sum_error_model = sum_error_model + abs(ht_neige_naive_model - data_batch.loc[i, ['hauteur_neige']].item())
        
    error_model = sum_error_model/(data_batch.shape[0] - 1)
    error_list_naive_model.append(error_model)
    
    #plot and save
    plt.figure(figsize=(20, 10) )
    plt.plot(data_batch.date, data_batch.hauteur_neige, label = 'hauteur neige')
    plt.plot(data_batch.date, data_batch.ht_neige_naive_model, 
             label = 'hauteur neige modele naif, mean error =' + str(error_model)[0:4])
    plt.legend()
    plt.title(data_batch.batch.unique()[0])
    plt.grid(True)
    plt.savefig('figures/predict_naive_model/' + batch + '.png')
    plt.show()


error_list_naive_model











################ PREDICTION SUR BATCHES TEST des modeles : RNN, LSTM, GRU ######## -pred, plot and save-

#initialisation tables erreurs
list_mean_diff_simplernn_test = []
list_mean_diff_gru_test = []
list_mean_diff_lstm_test = []
list_mean_diff_cnn_test = []

#valeurs y prediction
yhat_model_lstm_20_test = model_lstm_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))
yhat_model_gru_20_test = model_gru_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))
yhat_model_simplernn_20_test = model_simplernn_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))
yhat_model_cnn_test = model_cnn.predict(test_x, batch_size = nb_batches - len(train_batch_select))


list_batches_test = np.delete(list_batches, train_batch_select)

for batch in range(0, len(test_x)):
    
    
    #zeros de chaque sequence, pour recalibration au jour 0
    zero_real = max_y*test_y[batch][0]
    zero_simplernn = max_y*yhat_model_simplernn_20_test[batch][0]
    zero_gru = max_y*yhat_model_gru_20_test[batch][0]
    zero_lstm = max_y*yhat_model_lstm_20_test[batch][0]
    zero_cnn = max_y*yhat_model_cnn_test[batch][0]
    
    
    #difference moyenne entre predit et reel (abs)
    mean_diff_simplernn = (sum(abs(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    mean_diff_gru = (sum(abs(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    mean_diff_lstm = (sum(abs(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    mean_diff_cnn = (sum(abs(max_y*yhat_model_cnn_test[batch] - zero_cnn + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    
   
    #ecart type de la difference entre predit et reel (abs)
    std_diff_simplernn = np.std(abs(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real - max_y*test_y[batch]))
    std_diff_gru = np.std(abs(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real - max_y*test_y[batch]))
    std_diff_lstm = np.std(abs(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real - max_y*test_y[batch]))
    std_diff_cnn = np.std(abs(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real - max_y*test_y[batch]))
   
    
    #plot
    plt.figure(figsize= (20,10))
    plt.plot(max_y*test_y[batch], label = 'reel') #blue
    plt.plot(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real, 
             label = 'simple RNN 20, diff moy ' + str(mean_diff_simplernn)[0:4] + ' metres, ecart-type diff ' + str(std_diff_simplernn)[0:4])
    plt.plot(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real, 
             label = 'GRU 20, diff moy ' + str(mean_diff_gru)[0:4] + ' metres, ecart-type diff ' + str(std_diff_gru)[0:4])
    plt.plot(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real, 
             label = 'LSTM 20, diff moy ' + str(mean_diff_lstm)[0:4] + ' metres, ecart-type diff ' + str(std_diff_lstm)[0:4])
    plt.plot(max_y*yhat_model_cnn_test[batch] - zero_cnn + zero_real, 
             label = 'CNN, diff moy ' + str(mean_diff_cnn)[0:4] + ' metres, ecart-type diff ' + str(std_diff_cnn)[0:4])
    plt.title(list_batches_test[batch])
    plt.legend()
    plt.grid(True)
    #plt.savefig('figures/predict_compare_gru_rnn_lstm_with_recalibration_zero/test/' + list_batches_test[batch] + '.png')
    plt.show()
    
    
    #list erreurs
    list_mean_diff_simplernn_test.append(mean_diff_simplernn)
    list_mean_diff_gru_test.append(mean_diff_gru)
    list_mean_diff_lstm_test.append(mean_diff_lstm)
    list_mean_diff_cnn_test.append(mean_diff_cnn)





################ PREDICTION SUR BATCHES TRAIN : RNN, LSTM, GRU ########## -pred, plot and save-

#initialisation tables erreurs
list_mean_diff_simplernn_train = []
list_mean_diff_gru_train = []
list_mean_diff_lstm_train = []

#valeurs y prediction
yhat_model_lstm_20_train = model_lstm_20.predict(train_x, batch_size = len(train_batch_select))
yhat_model_gru_20_train = model_gru_20.predict(train_x, batch_size = len(train_batch_select))
yhat_model_simplernn_20_train = model_simplernn_20.predict(train_x, batch_size = len(train_batch_select))

#figures of train batches (90% of all batches)
list_batches_train = list_batches[train_batch_select]
for batch in range(0, len(train_x)):
    
    #zeros de chaque sequence, pour recalibration au jour 0
    zero_real = max_y*train_y[batch][0]
    zero_simplernn = max_y*yhat_model_simplernn_20_train[batch][0]
    zero_gru = max_y*yhat_model_gru_20_train[batch][0]
    zero_lstm = max_y*yhat_model_lstm_20_train[batch][0]
    
    #difference moyenne entre predit et reel (abs)
    mean_diff_simplernn = (sum(abs(max_y*yhat_model_simplernn_20_train[batch] - zero_simplernn + zero_real - max_y*train_y[batch]))/len_batch_sequence)[0]
    mean_diff_gru = (sum(abs(max_y*yhat_model_gru_20_train[batch] - zero_gru + zero_real - max_y*train_y[batch]))/len_batch_sequence)[0]
    mean_diff_lstm = (sum(abs(max_y*yhat_model_lstm_20_train[batch] - zero_lstm + zero_real - max_y*train_y[batch]))/len_batch_sequence)[0]
   
    #ecart type de la difference entre predit et reel (abs)
    std_diff_simplernn = np.std(abs(max_y*yhat_model_simplernn_20_train[batch] - zero_simplernn + zero_real - max_y*train_y[batch]))
    std_diff_gru = np.std(abs(max_y*yhat_model_gru_20_train[batch] - zero_gru + zero_real - max_y*train_y[batch]))
    std_diff_lstm = np.std(abs(max_y*yhat_model_lstm_20_train[batch] - zero_lstm + zero_real - max_y*train_y[batch]))
   
    #plot
    plt.figure(figsize= (20,10))
    plt.plot(max_y*train_y[batch], label = 'reel') #blue
    plt.plot(max_y*yhat_model_simplernn_20_train[batch] - zero_simplernn + zero_real, 
             label = 'simple RNN 20, diff moy ' + str(mean_diff_simplernn)[0:4] + ' metres, ecart-type diff ' + str(std_diff_simplernn)[0:4])
    plt.plot(max_y*yhat_model_gru_20_train[batch] - zero_gru + zero_real, 
             label = 'GRU 20, diff moy ' + str(mean_diff_gru)[0:4] + ' metres, ecart-type diff ' + str(std_diff_gru)[0:4])
    plt.plot(max_y*yhat_model_lstm_20_train[batch] - zero_lstm + zero_real, 
             label = 'LSTM 20, diff moy ' + str(mean_diff_lstm)[0:4] + ' metres, ecart-type diff ' + str(std_diff_lstm)[0:4])
    plt.title(list_batches_train[batch])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/predict_compare_gru_rnn_lstm_with_recalibration_zero/train/' + list_batches_train[batch] + '.png')
    plt.show()

    #list erreurs
    list_mean_diff_simplernn_train.append(mean_diff_simplernn)
    list_mean_diff_gru_train.append(mean_diff_gru)
    list_mean_diff_lstm_train.append(mean_diff_lstm)









#### PERFORMANCE ON TEST SET
    

model_lstm_20_evaluate = model_lstm_20.evaluate(train_x, train_y, 
                                                  batch_size= len(train_batch_select), verbose=1)

model_lstm_20_evaluate 



model_gru_20_evaluate = model_gru_20.evaluate(train_x, train_y, 
                                                  batch_size= len(train_batch_select), verbose=1)

model_gru_20_evaluate 


model_simplernn_20_evaluate = model_simplernn_20.evaluate(train_x, train_y, 
                                                  batch_size= len(train_batch_select), verbose=1)

model_simplernn_20_evaluate 
