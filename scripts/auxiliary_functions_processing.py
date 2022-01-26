# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:41:31 2020

@author: ichid
"""

import math
import pandas
import numpy
import datetime

#fonction ajout rayonnement solaire
def rayonnement_solaire(latitude, jour):
    
    
    #maximum et minimum rayonnement possible sur Terre en tout point. 23,5 = inclinaison axe Terre aujourd'hui
    max_ray = math.cos((latitude - 23.5) *2 *math.pi/360)
    min_ray = math.cos((latitude + 23.5) *2 *math.pi/360)
    
    #conversion date en jour dans l'annee
    #month_extracted = int(str(jour)[4:6])
    #daym_extracted = int(str(jour)[6:8])
    month_extracted = jour.month
    daym_extracted = jour.day
    num_jour = (month_extracted - 1)*30.5 + daym_extracted
    
    #calcul rayonnement en fonction du jour dans l'annee et de la latitude
    ray = (max_ray + min_ray)/2 + math.cos(2*math.pi*(num_jour - 172)/365.25 )*(max_ray - min_ray)/2
    
    return ray


def processing_liste_stations(liste_stations):
    
    #correction des noms : majuscules, termes indésirables, etc..
    for i in range(0, liste_stations.shape[0]):
        #noms indésirables (Nivo etc..), mise en minuscules
        nom_sta = liste_stations.loc[i, ['Nom']].item()
    
        #tous les "nivo" dans le nom enleves
        nom_sta = nom_sta.replace('-NIVO', '')
        nom_sta = nom_sta.replace('_NIVO', '')
        nom_sta = nom_sta.replace(' NIVO', '')
    
        #tous les "st" deviennent "Saint"
        nom_sta = nom_sta.replace('ST ', 'Saint ')
        nom_sta = nom_sta.replace('St ', 'Saint ')
        
        #tous les "AUXI" deviennent "Saint"
        nom_sta = nom_sta.replace(' AUXI', '')
    
        #mise en majuscule des premieres lettres uniquement
        nom_sta = nom_sta.lower().title()
    
        liste_stations.loc[i, ['Nom']] = nom_sta


    #ajout categorie massif : Pyrenees, Massif Central, 
    #Alpes du Nord, Alpes du sud, Corse, Jura, Vosges
    liste_stations['massif'] = numpy.nan
    
    liste_stations.loc[(liste_stations.Latitude < 44.0) & (liste_stations.Longitude < 4.0), "massif"] = "Pyrenees"
    liste_stations.loc[(liste_stations.Latitude < 47.0) & (liste_stations.Latitude > 46.5), "massif"] = "Jura"
    liste_stations.loc[(liste_stations.Latitude > 47.0), "massif"] = "Vosges"
    liste_stations.loc[(liste_stations.Latitude < 43.0) & (liste_stations.Longitude > 7.0), "massif"] = "Corse"
    liste_stations.loc[(liste_stations.Latitude > 44.0) & (liste_stations.Longitude < 4.8), "massif"] = "Massif Central"
    liste_stations.loc[(liste_stations.Latitude > 44.85) & (liste_stations.Latitude < 46.5) & (liste_stations.Longitude > 4.8), "massif"] = "Alpes du Nord"
    liste_stations.loc[(liste_stations.Latitude > 43.0) & (liste_stations.Latitude < 44.85) & (liste_stations.Longitude > 4.8), "massif"] = "Alpes du Sud"

    liste_stations.loc[(liste_stations.Nom == "La Grave - La Meije"), "massif"] = "Alpes du Sud"
    liste_stations.loc[(liste_stations.Nom == "Le Monetier"), "massif"] = "Alpes du Sud"
    liste_stations.loc[(liste_stations.Nom == "Serre Chevalier"), "massif"] = "Alpes du Sud"
    liste_stations.loc[(liste_stations.Nom == "Montgenevre-Le Chalvet"), "massif"] = "Alpes du Sud"
    liste_stations.loc[(liste_stations.Nom == "Pelvoux Saint Antoine"), "massif"] = "Alpes du Sud"
    liste_stations.loc[(liste_stations.Nom == "Nevache-Buffere"), "massif"] = "Alpes du Sud"


    return(liste_stations)
    
    
    

#fonction remplissage trous hauteur_neige
#il faut isoler le batch occurent des autres avant de faire l'application. 
#pour ne pas qu'avant la date min(15 decembre) ou apres la date max (15 avril) on tombe sur un autre batch
#def remplissage_hauteur_neige(dataframe):
    
    #cas ou entre deux 
    
    
    #cas ou debut de batch et rien avant, plusieurs cas
    
    
    
    #cas ou au
def filling_holes_data(data_with_holes, list_batches):
    
    data_filled = pandas.DataFrame()
    
    for batch in list_batches:

        data_tampon = data_with_holes[data_with_holes.batch == batch]


        data_tampon['_vent_moy_10min_m/s'] = data_tampon['_vent_moy_10min_m/s'].fillna(0)
    
        data_tampon['temperature'] = data_tampon['temperature'].interpolate(method = 'linear', 
                   limit_direction = 'both')
    
        #data_tampon['point_de_rosee'] = data_tampon['point_de_rosee'].interpolate(method = 'linear', 
        #           limit_direction = 'both')
    
        #data_tampon['humidite'] = data_tampon['humidite'].interpolate(method = 'linear', 
        #       limit_direction = 'both')
    
        data_tampon['nebulosite'] = data_tampon['nebulosite'].fillna(5)
    
        data_tampon['nebulosite_etage_inf'] = data_tampon['nebulosite_etage_inf'].fillna(5)
    
        data_tampon['precipitations_24h'] = data_tampon['precipitations_24h'].fillna(0)
    
        data_tampon['temperature_min_24h'] = data_tampon['temperature_min_24h'].interpolate(method = 'linear', 
                   limit_direction = 'both')
    
        data_tampon['temperature_max_24h'] = data_tampon['temperature_max_24h'].interpolate(method = 'linear', 
               limit_direction = 'both')
    
        data_tampon['hauteur_neige'] = data_tampon['hauteur_neige'].interpolate(method = 'linear', 
                   limit_direction = 'both')
    
        data_tampon['hauteur_neige_fraiche'] = data_tampon['hauteur_neige_fraiche'].fillna(0)
        
        data_tampon['rayonnement_solaire'] = data_tampon['rayonnement_solaire'].interpolate(method = 'linear', 
                   limit_direction = 'both')
        
        print(batch)
    
        data_filled = data_filled.append(data_tampon)
        
    return(data_filled)