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

### fin de la reouverture des donnees et modeles necessaires !

#recalcul du nb de batchs et de leur longueur
nb_batches = len(list_batches)
len_batch_sequence = dataset_batches_x.shape[1]

#reexecution des fonctions de constitution des train_x, test_x, train_y, test_y
train_x = dataset_batches_x[train_batch_select]
test_x = np.delete(dataset_batches_x, train_batch_select, axis = 0)
train_y = dataset_batches_y[train_batch_select]
test_y = np.delete(dataset_batches_y, train_batch_select, axis = 0)






################ PREDICT ON TEST BATCHES ##### -pred, plot and save-

list_mean_diff_simplernn_test = []
list_mean_diff_gru_test = []
list_mean_diff_lstm_test = []

yhat_model_lstm_20_test = model_lstm_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))
yhat_model_gru_20_test = model_gru_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))
yhat_model_simplernn_20_test = model_simplernn_20.predict(test_x, batch_size = nb_batches - len(train_batch_select))

#figures of test batches (10% of all batches)
list_batches_test = np.delete(list_batches, train_batch_select)
for batch in range(0, len(test_x)):
    
    #zeros of each sequence (for recalibrating)
    zero_real = max_y*test_y[batch][0]
    zero_simplernn = max_y*yhat_model_simplernn_20_test[batch][0]
    zero_gru = max_y*yhat_model_gru_20_test[batch][0]
    zero_lstm = max_y*yhat_model_lstm_20_test[batch][0]
    
    #difference moyenne entre predit et reel (abs)
    mean_diff_simplernn = (sum(abs(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    mean_diff_gru = (sum(abs(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
    mean_diff_lstm = (sum(abs(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real - max_y*test_y[batch]))/len_batch_sequence)[0]
   
    #ecart type de la difference entre predit et reel (abs)
    std_diff_simplernn = np.std(abs(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real - max_y*test_y[batch]))
    std_diff_gru = np.std(abs(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real - max_y*test_y[batch]))
    std_diff_lstm = np.std(abs(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real - max_y*test_y[batch]))
   
    #plot
    plt.figure(figsize= (20,10))
    plt.plot(max_y*test_y[batch], label = 'reel') #blue
    plt.plot(max_y*yhat_model_simplernn_20_test[batch] - zero_simplernn + zero_real, 
             label = 'simple RNN 20, diff moy ' + str(mean_diff_simplernn)[0:4] + ' metres, ecart-type diff ' + str(std_diff_simplernn)[0:4])
    plt.plot(max_y*yhat_model_gru_20_test[batch] - zero_gru + zero_real, 
             label = 'GRU 20, diff moy ' + str(mean_diff_gru)[0:4] + ' metres, ecart-type diff ' + str(std_diff_gru)[0:4])
    plt.plot(max_y*yhat_model_lstm_20_test[batch] - zero_lstm + zero_real, 
             label = 'LSTM 20, diff moy ' + str(mean_diff_lstm)[0:4] + ' metres, ecart-type diff ' + str(std_diff_lstm)[0:4])
    plt.title(list_batches_test[batch])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/predict_compare_gru_rnn_lstm_with_recalibration_zero/test/' + list_batches_test[batch] + '.png')
    plt.show()
    
    list_mean_diff_simplernn_test.append(mean_diff_simplernn)
    list_mean_diff_gru_test.append(mean_diff_gru)
    list_mean_diff_lstm_test.append(mean_diff_lstm)





##### PREDICT ON TRAIN BATCHES ##### -pred, plot and save-

list_mean_diff_simplernn_train = []
list_mean_diff_gru_train = []
list_mean_diff_lstm_train = []

yhat_model_lstm_20_train = model_lstm_20.predict(train_x, batch_size = len(train_batch_select))
yhat_model_gru_20_train = model_gru_20.predict(train_x, batch_size = len(train_batch_select))
yhat_model_simplernn_20_train = model_simplernn_20.predict(train_x, batch_size = len(train_batch_select))

#figures of train batches (90% of all batches)
list_batches_train = list_batches[train_batch_select]
for batch in range(0, len(train_x)):
    
    #zeros of each sequence (for recalibrating)
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
