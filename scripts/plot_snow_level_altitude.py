# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:48:55 2020

@author: ichid
"""

import pandas as pd
import matplotlib.pyplot as plt

nivo_complete_morning = pd.read_excel('D:/1Data_science&Big_data/Exercices/Montagne_Meteo_France/data/output/table_complete_morning.xlsx')




alps = nivo_complete_morning[nivo_complete_morning.date == dt.datetime(2018, 1, 30).date()]
alps = alps[alps.Longitude > 5.5]
alps = alps[alps.Latitude > 44.5]
alps = alps[(alps.hauteur_neige != 'mq')]
alps['hauteur_neige'] = pd.to_numeric(alps['hauteur_neige'])

plt.figure(figsize=(18,15))
alps_x = alps['Altitude'].values
alps_y = alps['hauteur_neige'].values
alps_nom = alps['Nom'].values

plt.plot(alps_x, alps_y, 'ro')
for i, txt in enumerate(alps_nom):
    plt.annotate(s = txt, 
                 xy = (alps_x[i], alps_y[i]),
                 textcoords = "offset points",
                 xytext=(0,10))
    
plt.show()