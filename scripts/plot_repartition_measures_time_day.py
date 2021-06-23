# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:38:07 2020

@author: ichid
"""

import pandas as pd
import matplotlib.pyplot as plt

nivo_complete = pd.read_excel('D:/1Data_science&Big_data/Exercices/Montagne_Meteo_France/data/output/table_complete.xlsx')


#repartition des observations par heure de la journee pour toutes les observations
date_str_current = nivo_complete['datetime'].tolist()                         
time_str = date_str_current

list_perssfrai = nivo_complete['perssfrai'].to_list()

for i in range(0, len(date_str_current)):
    time_str[i] = date_str_current[i].hour + (date_str_current[i].minute)/60



plt.figure(figsize = (20, 10))
plt.hist(x = time_str, bins = 96)
plt.grid(True)
plt.show()




### comptage nb de mesures pour une meme station dans la meme journee
count_sta_day = nivo_complete.groupby(['Nom', 'date']).count()['Latitude']


