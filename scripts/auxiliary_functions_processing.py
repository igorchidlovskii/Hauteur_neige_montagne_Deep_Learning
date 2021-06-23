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