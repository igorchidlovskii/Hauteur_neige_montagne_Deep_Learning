# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:46:57 2021

@author: ichid
"""

#plt.figure(figsize=(15, 15))

import matplotlib.pyplot as plt


scatter_x = liste_stations.Latitude.to_list()
scatter_y = liste_stations.Longitude.to_list()

group = liste_stations.massif.to_list()

group_colors = {'Alpes du Nord':'red', 'Alpes du Sud':'orange', 'Jura': 'green', 
                'Pyrenees':'blue', 'Corse':'black', 'Vosges':'yellow', 'Massif Central':'purple'}

plt.scatter(x = scatter_x, y = scatter_y, c =  liste_stations.massif.map(group_colors).to_list() )
plt.show()





nivo_test = nivo_complete_morning[nivo_complete_morning.date == dt.date.today()-dt.timedelta(days=1)]
nivo_test = nivo_test[nivo_test.hauteur_neige != 'mq']
#nivo_test = nivo_test.merge(right = liste_stations[['Nom', 'massif']], how = 'left',
 #                           on = 'Nom')

x_coords = nivo_test.Altitude.to_list()
y_coords = nivo_test.hauteur_neige.to_list()
y_coords1 = [float(x) for x in y_coords]
types = nivo_test.Nom.to_list()

colors_mapping = {'Alpes du Nord':'r', 'Alpes du Sud':'c', 'Jura': 'g', 
                'Pyrenees':'b', 'Corse':'k', 'Vosges':'y', 'Massif Central':'m'}

group_massif = nivo_test.massif.to_list()
group_colors = nivo_test.massif.map(colors_mapping).to_list()

plt.figure(figsize = (20, 15))
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords1[i]
    plt.scatter(x, y,  c= group_colors[i], label = group_massif[i])
    plt.text(x, y, type, fontsize=9)
    #plt.legend()
    plt.xlabel('Altitude point de mesure')
    plt.ylabel('Hauteur de neige')
    plt.title('Enneigement 24 decembre 2021 France')
    plt.grid(True)

    
    
    
    
