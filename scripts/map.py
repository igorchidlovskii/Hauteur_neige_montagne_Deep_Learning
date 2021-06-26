# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:51:37 2021

@author: ichid
"""

### tracage de la carte 

import os
import pandas as pd
import folium

os.chdir('D:/Documents/Travail perso/Exercices/Montagne_Meteo_France')

#import liste stations
liste_stations = pd.read_csv('data/input/listes_stations.csv', sep = ';')

stations_map = folium.Map(location=[45,1], zoom_start=5)

stations_map.add_child(folium.GeoJson('data/input/postesNivo.json'))

# Display the map -> le display ne marche pas : il fonctionne que sur Jupyter Notebook 
#copiez-collez sur Jupyter Notebook ce script et decommentez la ligne ci-dessous pour voir la carte directement
#display(stations_map)

#sinon enregistrement en .html et ouverture dans un navigateur
stations_map.save('maps_stations.html')
