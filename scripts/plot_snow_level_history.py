# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:44:30 2020

@author: ichid

"""

import pandas as pd
import matplotlib.pyplot as plt

#nivo_complete_morning = pd.read_excel('D:/1Data_science&Big_data/Exercices/Montagne_Meteo_France/data/output/table_complete.xlsx')

for sta in nivo_complete_morning.Nom.unique():

    year_begin = 1996
    year_end = 2020

    one_sta = nivo_complete_morning[nivo_complete_morning.Nom == sta]
    one_sta = one_sta[one_sta.hauteur_neige != "mq"]
    one_sta['hauteur_neige'] = pd.to_numeric(one_sta['hauteur_neige'])
 
    plt.figure(figsize=(20, 100))
    plot_level = 1

    for y in range(year_begin, year_end +1):
        plt.subplot(len(range(year_begin,   year_end+1)), 1, plot_level)
        plt.plot(one_sta['date'], one_sta['hauteur_neige'], 'ro')
        plt.axis([dt.datetime(y,11,1,0,0,0), dt.datetime(y+1,6,1,0,0,0), 0, 3])
        plt.grid(True)
        plt.ylabel('hauteur neige ' + str(y) + ' '+ str(y+1) )
        plot_level = plot_level + 1
 
    #enregistrement image
    alt_one_sta = one_sta['Altitude'].unique()
    plt.savefig('figures/hauteur_neige/'+ sta +'.png')
    plt.show()

    #affichage nom + altitude station
    print(one_sta['Nom'].unique(), one_sta['Altitude'].unique()) 







##### affichage une station specifique (graphiques les uns sous les autres)
    
year_begin = 1996
year_end = 2020

one_sta = nivo_complete_morning[nivo_complete_morning.Nom == 'Iraty']
one_sta = one_sta[one_sta.hauteur_neige != "mq"]
one_sta['hauteur_neige'] = pd.to_numeric(one_sta['hauteur_neige'])
 
plt.figure(figsize=(20, 100))
plot_level = 1

for y in range(year_begin, year_end +1):
    plt.subplot(len(range(year_begin,   year_end+1)), 1, plot_level)
    plt.plot(one_sta['date'], one_sta['hauteur_neige'], 'ro')
    plt.axis([dt.datetime(y,11,1,0,0,0), dt.datetime(y+1,6,1,0,0,0), 0, 3])
    plt.grid(True)
    plt.ylabel('hauteur neige ' + str(y) + ' '+ str(y+1) )
    plot_level = plot_level + 1
 
#enregistrement image
alt_one_sta = one_sta['Altitude'].unique()
#plt.savefig('figures/hauteur_neige/'+ sta +'.png')
plt.show()







#tests
one_sta_2020 = one_sta[one_sta['datetime'] >dt.datetime(2019,11,1,0,0,0)]
one_sta_2020 = one_sta_2020[one_sta_2020['datetime'] <dt.datetime(2020,6,1,0,0,0)]
one_sta_2021 = one_sta[one_sta['datetime'] >dt.datetime(2020,11,1,0,0,0)]
one_sta_2021 = one_sta_2021[one_sta_2021['datetime'] <dt.datetime(2021,6,1,0,0,0)]



fig, ax = plt.subplots()

ax.plot(one_sta_2020['date'], one_sta_2020['hauteur_neige'], 'b.')
ax.plot(one_sta_2021['date'], one_sta_2021['hauteur_neige'], 'b.')
ax.grid(True)



### showing all graphics combined in one
year_begin = 1996
year_end = 2020

one_sta = nivo_complete_morning[nivo_complete_morning.Nom == 'La Plagne']
one_sta = one_sta[one_sta.hauteur_neige != "mq"]
one_sta['hauteur_neige'] = pd.to_numeric(one_sta['hauteur_neige'])

#plt.figure(figsize=(20, 8))
#plot_level = 1
fig, ax = plt.subplots()

for y in range(year_begin, year_end + 1):
    one_sta_f = one_sta[one_sta['datetime'] >dt.datetime(y,11,1,0,0,0) & one_sta['datetime'] <dt.datetime(y+1,6,1,0,0,0)]
    one_sta_f['date_recal'] = one_sta_f['date'].apply(lambda x: x.year) 
    




