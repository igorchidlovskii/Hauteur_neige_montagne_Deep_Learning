# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

"""

#import packages
import pandas as pd
import numpy as np
import datetime as dt
import os
#import csv
#import folium


### parametres temporels pour nos batchs
annee_min = 1996
annee_max = 2020
mois_min = 12
jour_min = 14
mois_max = 4
jour_max = 16

#pour quelle fraction minimum de completude sur la periode (actuellement 15 dec-15 avr) on l'accepte et on le met comme batch pour l'entrainement ? 
completude_pour_batch = 0.8

#veut-on enregistrer les tables intermediaires en Excel ?
write_excel_cleaning = False

#veut-on extraire depuis le site Meteo-France les dernieres donnees (jusqu'a la veille)
update_last_data = False

#placement dossier de travail
os.chdir('D:/Documents/Travail perso/Exercices/Montagne_Meteo_France/scripts')

#import fonctions
import auxiliary_functions_cleaning
import auxiliary_functions_processing
import extract

os.chdir('D:/Documents/Travail perso/Exercices/Montagne_Meteo_France')

#import liste stations
liste_stations = pd.read_csv('data/input/listes_stations.csv', sep = ';')

#extraction depuis le site de Meteo-France des dernieres donnees, si l'on veut
if update_last_data == True:
    extract.extract_last_data()

#import des données nivologiques: csv les uns après les autres, mois par mois
files_list = os.listdir('data/input/nivo')
nivo = pd.read_csv('data/input/nivo/' + files_list[1], sep = ';')
for i in files_list:
    nivo_int = pd.read_csv(('data/input/nivo/' + i) , sep = ';')
    nivo = nivo.append(nivo_int, ignore_index = True)
    print(i)

print('import donnees terminé')
# on a maintenant toutes les données brutes 

nivo_raw = nivo.copy()    





################################################### CLEANING ###############################################################

#Nettoyage des lignes avec valeurs aberrantes (nom du titre dans la table)
#nivo = nivo.drop(labels = ['numer_sta'], axis = 1)
if ('numer_sta' in nivo):
    nivo = nivo[nivo.numer_sta != 'numer_sta']

nivo['numer_sta'] = pd.to_numeric(nivo['numer_sta'])

#conversion date au bon format
# PASSAGE EN NUMPY (des milliers de fois plus rapide que la boucle for classique)
datetime_str = nivo['date'].values
for i in range(0, len(datetime_str)):    
    datetime_str[i] = dt.datetime.strptime(str(datetime_str[i]), "%Y%m%d%H%M%S")
    
#repassage de la colonne date en Series pour aller dans la table pandas
nivo['datetime'] = pd.Series(datetime_str)
nivo['date1'] = nivo['datetime'].apply(lambda x: x.date() )
nivo = nivo.drop(columns = ['date'])
nivo['hour'] = nivo['datetime'].apply(lambda x: x.hour )
nivo['year'] = nivo['datetime'].apply(lambda x: x.year )

nivo = nivo[['numer_sta', 'datetime', 'date1', 'haut_sta', 'dd', 'ff', 't', 'td', 'u', 'ww', 'w1',
       'w2', 'n', 'nbas', 'hbas', 'cl', 'cm', 'ch', 'rr24', 'tn12', 'tn24',
       'tx12', 'tx24', 'ht_neige', 'ssfrai', 'perssfrai', 'phenspe1',
       'phenspe2', 'nnuage1', 't_neige', 'etat_neige', 'prof_sonde',
       'nuage_val', 'chasse_neige', 'aval_descr', 'aval_genre', 'aval_depart',
       'aval_expo', 'aval_risque', 'dd_alti', 'ff_alti', 'ht_neige_alti',
       'neige_fraiche', 'teneur_eau', 'grain_predom', 'grain_nombre',
       'grain_diametr', 'homogeneite', 'm_vol_neige', 'Unnamed: 48',
       'Unnamed: 49', 'hour', 'year']]

nivo = nivo.rename(columns = {'date1': 'date'})

#supression des lignes avec date nulle
nivo = nivo.dropna(subset=['datetime'])

#supression doublons (il y en a un peu)
nivo = nivo.drop_duplicates()

print(nivo.count())
print('ok doublons - dates vides')

#Merge avec liste des stations (avec altitude, latitute, longitude, nom) avec en clef de merge le station
nivo_complete = pd.merge(left = liste_stations, right = nivo, how = 'outer', 
                         left_on = 'ID', right_on = 'numer_sta')

nivo_complete.Nom.value_counts().to_excel('data/output/count_stations.xlsx')

#selection colonnes qui nous interessent
nivo_complete = nivo_complete[['Latitude', 'Longitude', 'Altitude', 'Nom', 'numer_sta', 'date', 'datetime','ff', 
                               't', 'td', 'u', 'n', 'nbas', 'rr24', 'tn24', 'tx24', 'ht_neige', 'ssfrai',  
                               'perssfrai', 'hour', 'year']]

#on supprime les periode_mesure_neige_fraiche 50, qui sont souvent des donnees manquantes ou tres incompletes
pd.crosstab(nivo_complete['year'], nivo_complete['perssfrai'], dropna = False)
nivo_complete.perssfrai = nivo_complete.perssfrai.apply(str)
nivo_complete.perssfrai.value_counts()
nivo_complete = nivo_complete[nivo_complete.perssfrai != '-50']


#affichage pourcentage de valeurs manquantes
print("% de valeurs manquantes pour :")
print("vitesse du vent:", 100*len(nivo_complete[nivo_complete.ff == "mq"])/len(nivo_complete.ff))
print("temperature actuelle:", 100*len(nivo_complete[nivo_complete.t == "mq"])/len(nivo_complete.t))
print("point de rosee:", 100*len(nivo_complete[nivo_complete.td == "mq"])/len(nivo_complete.td))
print("humidite:", 100*len(nivo_complete[nivo_complete.u == "mq"])/len(nivo_complete.u))
print("nebulosite:", 100*len(nivo_complete[nivo_complete.n == "mq"])/len(nivo_complete.n))
print("nebulosite etage inf:", 100*len(nivo_complete[nivo_complete.nbas == "mq"])/len(nivo_complete.nbas))
print("precipitations dernières 24h:", 100*len(nivo_complete[nivo_complete.rr24 == "mq"])/len(nivo_complete.rr24), 
      "-> mettre zero dans les trous ?")
print("temperature min 24h:", 100*len(nivo_complete[nivo_complete.tn24 == "mq"])/len(nivo_complete.tn24), 
     "-> interpolation dans les trous ?")
print("temperature max 24h:", 100*len(nivo_complete[nivo_complete.tx24 == "mq"])/len(nivo_complete.tx24),
     "-> interpolation dans les trous ?")
print("hauteur neige:", 100*len(nivo_complete[nivo_complete.ht_neige == "mq"])/len(nivo_complete.ht_neige))
print("hauteur neige fraiche:", 100*len(nivo_complete[nivo_complete.ssfrai == "mq"])/len(nivo_complete.ssfrai))
print("periode neige fraiche:", 100*len(nivo_complete[nivo_complete.perssfrai == "mq"])/len(nivo_complete.perssfrai))

#on renomme les noms de colonnes
nivo_complete = nivo_complete.rename(columns = {'ff':'_vent_moy_10min_m/s', 
                                                't':'temperature', 
                                                'td':'point_de_rosee', 
                                                'u':'humidite', 
                                                'n':'nebulosite', 
                                                'nbas':'nebulosite_etage_inf', 
                                                '_vent_moy_10min_m/s':'vent_moy',
                                                'rr24':'precipitations_24h', 
                                                'tn24': 'temperature_min_24h', 
                                                'tx24':'temperature_max_24h',
                                                'ht_neige':'hauteur_neige', 
                                                'ssfrai':'hauteur_neige_fraiche', 
                                                'perssfrai':'periode_mesure_neige_fraiche'} )


nivo_complete = nivo_complete.dropna(subset=['date'])
#nivo_complete['date_raw'] = nivo_complete['date_raw'].astype(str)

nivo_complete = nivo_complete.dropna(subset=['Nom'])

#on ordonne la table : d'abord les stations, ensuite par date
nivo_complete = nivo_complete.sort_values(by=['Nom', 'date'], ascending = True)

#on enregistre la table complete si l'on veut
if (write_excel_cleaning == True):
    nivo_complete.to_excel('data/output/table_complete.xlsx')






################################################### CREATION DES BATCHES ###############################################################
#on supprime les valeurs dont l'enregistrement a lieu apres 10h du matin
nivo_complete_morning = nivo_complete[nivo_complete['hour'] < 10]

#supression vieilles tables (car sont massives et prenent de la RAM !)
del[nivo,
    nivo_raw,
    datetime_str,
    nivo_complete]

if (write_excel_cleaning == True):
    #on enregistre la table des matins
    nivo_complete_morning.to_excel('data/output/table_complete_morning.xlsx')

#liste des stations reportees
liste_stations_reporte = nivo_complete_morning['Nom'].unique()

#completude par station et par periode temporelle selectionnee
completude_station_periode = auxiliary_functions_cleaning.completude_station_and_period(nivo = nivo_complete_morning, 
                                                                       liste_stations_reporte = liste_stations_reporte, 
                                                                       save_excel = False,
                                                                       annee_min = annee_min, annee_max = annee_max, 
                                                                       mois_min = mois_min, jour_min = jour_min, 
                                                                       mois_max = mois_max, jour_max = jour_max)

if (write_excel_cleaning == True):
    #on enregistre la table des completudes
    completude_station_periode.to_excel('data/output/completude_station_periode_15dec_15avr.xlsx')



#on regarde les dates pour une station ou il y a plus d'une mesure () 
#(inutile car par annee, completude_station_periode est mieux et prend le relais pour les periodes specifiques)
#il va falloir ensuite les traiter
count_dates = nivo_complete_morning.groupby(['Nom', 'date']).apply(lambda x : x.count()).filter(['Latitude'])
pd.set_option('display.max_rows', count_dates.shape[0]+1)
count_dates[count_dates.Latitude > 1]

#selection de la liste des batchs, en exigeant une certaine completude pour les donnees dans la petiode batch, et 
completude_station_periode_15dec_15avr_80 = completude_station_periode[\
                                                        (completude_station_periode['completude_mesure'] > completude_pour_batch) &
                                                        (completude_station_periode['temperature_max_24h'] > 0.7) & 
                                                        (completude_station_periode['temperature_min_24h'] > 0.7) & 
                                                        (completude_station_periode['hauteur_neige_fraiche'] > completude_pour_batch)]


completude_station_periode_15dec_15avr_80['periode'] = completude_station_periode_15dec_15avr_80['periode'].astype(int)

nivo_complete_morning_cleaned = auxiliary_functions_cleaning.delete_bad_lines_and_to_batch(table_stations_periods_filter = completude_station_periode_15dec_15avr_80, 
                                                                                           nivo = nivo_complete_morning,
                                                                                           mois_min = mois_min, jour_min = jour_min, 
                                                                                           mois_max = mois_max, jour_max = jour_max)

if (write_excel_cleaning == True):
    nivo_complete_morning_cleaned.to_excel('data/output/nivo_complete_morning_batchs.xlsx')



#on cree la table, qu'on copie, on filtre sur les stations reellement presentes au moins une fois
liste_stations_dates = liste_stations.copy()
liste_stations_presentes = list(dict.fromkeys(nivo_complete_morning['numer_sta'].to_list()))
liste_stations_dates = liste_stations_dates[liste_stations_dates['ID'].isin(liste_stations_presentes)]

#table toutes dates toutes stations vides (dessus on viendra merger notre table avec les mesures)
table_all_dates_all_stations = auxiliary_functions_cleaning.creation_table_all_stations_all_dates(date_min_interval = min(nivo_complete_morning['date']), 
                                  date_max_interval = max(nivo_complete_morning['date']),
                                  liste_stations = liste_stations_dates)

#passage en batch 
table_all_dates_all_stations_batch = auxiliary_functions_cleaning.to_batch_tables(table = table_all_dates_all_stations, 
                                                       table_batch = completude_station_periode_15dec_15avr_80,
                                                        mois_min = mois_min, jour_min = jour_min, 
                                                        mois_max = mois_max, jour_max = jour_max)

#merge entre table vide toutes stations - et table avec mesures
table_all_dates_all_stations_batch = table_all_dates_all_stations_batch.merge(right = nivo_complete_morning_cleaned,
                                                 on = ['Nom', 'date'], how = 'left')
    
#drop colonnes inutiles
table_all_dates_all_stations_batch = table_all_dates_all_stations_batch.drop(['Latitude_y', 'Longitude_y', 
                                            'Altitude_y', 'numer_sta', 'periode_mesure_neige_fraiche', 'year'], 
                                                axis = 1)

#noms batch : remplir trous
table_all_dates_all_stations_batch['batch'] = table_all_dates_all_stations_batch.\
                        apply(lambda x: (x['Nom'] + "_" + str(x['date'].year))
                        if (x['date'].month > 6)
                        else (x['Nom'] + "_" + str(x['date'].year-1) ), axis = 1)
      
        
list_batch_keep_compl = table_all_dates_all_stations_batch.groupby('batch').count()
list_batch_keep_compl2 = list_batch_keep_compl[list_batch_keep_compl.Latitude_x == max(list_batch_keep_compl.Latitude_x)].index
table_all_dates_all_stations_batch = table_all_dates_all_stations_batch[table_all_dates_all_stations_batch.batch.isin(list_batch_keep_compl2)]
        

if (write_excel_cleaning == True):
    table_all_dates_all_stations_batch.to_excel('data/output/table_complete_morning_cleaned_batchs.xlsx')
    
    
    
    
#on supprime les tables inutiles (qui prennent de la RAM !)
del [nivo_complete_morning_cleaned, 
     table_all_dates_all_stations,
     count_dates]
    


#on a maintenant une table des batchs, filtres sur les stations et periodes ou la completude des donnees etait de plus de 80%
#pour la suite il nous faut uniquement :
# 1 - table des batchs : table_all_dates_all_stations_batch
# 2 - table d'origine avec les matins : nivo_complete_morning

table_all_dates_all_stations_batch.count()


########################################################### PROCESSING ###############################################################

  

#passage en numerique des colonnes qui ne l'etaient pas (necessaire pour interpoler et pour la suite)
table_all_dates_all_stations_batch['Altitude_x'] = pd.to_numeric(table_all_dates_all_stations_batch['Altitude_x'])
table_all_dates_all_stations_batch['_vent_moy_10min_m/s'] = pd.to_numeric(table_all_dates_all_stations_batch['_vent_moy_10min_m/s'])
table_all_dates_all_stations_batch['temperature'] = pd.to_numeric(table_all_dates_all_stations_batch['temperature'])
table_all_dates_all_stations_batch['point_de_rosee'] = pd.to_numeric(table_all_dates_all_stations_batch['point_de_rosee'])
table_all_dates_all_stations_batch['humidite'] = pd.to_numeric(table_all_dates_all_stations_batch['humidite'])
table_all_dates_all_stations_batch['nebulosite'] = pd.to_numeric(table_all_dates_all_stations_batch['nebulosite'])
table_all_dates_all_stations_batch['nebulosite_etage_inf'] = pd.to_numeric(table_all_dates_all_stations_batch['nebulosite_etage_inf'])
table_all_dates_all_stations_batch['point_de_rosee'] = pd.to_numeric(table_all_dates_all_stations_batch['point_de_rosee'])
table_all_dates_all_stations_batch['precipitations_24h'] = pd.to_numeric(table_all_dates_all_stations_batch['precipitations_24h'])
table_all_dates_all_stations_batch['temperature_min_24h'] = pd.to_numeric(table_all_dates_all_stations_batch['temperature_min_24h'])
table_all_dates_all_stations_batch['temperature_max_24h'] = pd.to_numeric(table_all_dates_all_stations_batch['temperature_max_24h'])
table_all_dates_all_stations_batch['hauteur_neige_fraiche'] = pd.to_numeric(table_all_dates_all_stations_batch['hauteur_neige_fraiche'])

#rayonnement solaire : fonction puis interpolation
table_all_dates_all_stations_batch['rayonnement_solaire'] = table_all_dates_all_stations_batch.\
      apply(lambda x: auxiliary_functions_processing.rayonnement_solaire(x['Latitude_x'], x['datetime']),
            axis = 1)
    

#remplissage trous : modele le plus simple : interpolations ou remplissage par des zeros selon les parametres
table_all_dates_all_stations_batch_filled = auxiliary_functions_processing.filling_holes_data(
        data_with_holes = table_all_dates_all_stations_batch, 
                                list_batches = table_all_dates_all_stations_batch.batch.unique() )


if(write_excel_cleaning == True):
    table_all_dates_all_stations_batch_filled.to_excel('data/output/table_complete_morning_cleaned_batchs_filled.xlsx')

 

#### fin du main : prochains scripts : learning.py et predict.py

