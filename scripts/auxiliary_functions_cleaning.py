# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:26:29 2020

@author: ichid
"""

import numpy
import pandas
import datetime



#mois min = mois du debut de l'invervalle
#jour min = jour du debut de l'intervalle, qui va dans le mois ci-dessus
#mois max = mois de la fin de l'invervalle
#jour max = jour de la fin de l'intervalle, qui va dans le mois ci-dessus
def completude_station_and_period(nivo, liste_stations_reporte, save_excel, 
                                  annee_min, annee_max, mois_min, jour_min, mois_max, jour_max):
    
    completude_station_periode = pandas.DataFrame(index = range(0, len(liste_stations_reporte) ), 
                                                  columns = nivo.columns.drop(labels = ['Latitude', 
                                                                                'Longitude',  'Altitude', 'numer_sta', 
                                                                                'date', 'datetime', 'hour']))
    completude_station_periode['completude_mesure'] = numpy.nan
    completude_station_periode['station'] = pandas.Series(liste_stations_reporte)
    
    completude_station_periode_full = pandas.DataFrame(columns = nivo.columns.drop(labels = ['Latitude', 'Longitude',  'Altitude', 
                                                                                     'numer_sta', 'date', 'datetime', 'hour']))
    
    for y in range(annee_min, annee_max+1):
        completude_station_periode['periode'] = y    
        completude_station_periode_full = completude_station_periode_full.append(completude_station_periode)
    
    #pour chaque station dans la liste
    for sta in liste_stations_reporte: 
        #filtre sur la station
        nivo_sta = nivo[nivo["Nom"] == sta]
    
        for y in range(annee_min, annee_max+1):
            #filtre sur la periode temporelle du 15 dec au 15 avril
            nivo_occ = nivo_sta[(nivo_sta.date > datetime.datetime(y, mois_min, jour_min).date()) & 
                                (nivo_sta.date < datetime.datetime(y+1, mois_max, jour_max).date())]
        
            #calcul completude station
            interval_days = (datetime.datetime(y+1, mois_max, jour_max) - datetime.datetime(y, mois_min, jour_min)).days
            
            
            #calcul completude nb de mesures
            completude_station_periode_full.loc[(completude_station_periode_full.station == sta) & (completude_station_periode_full.periode == y), 
                                        ['completude_mesure'] ] = nivo_occ.shape[0]/interval_days
        
            #calcul completude par variable (mq + pas de mesure)
            for var in completude_station_periode_full.columns.drop(labels = ['station', 'completude_mesure', 'periode']) :
                completude_station_periode_full.loc[(completude_station_periode_full.station == sta) & (completude_station_periode_full.periode == y), 
                                               [var]] = (nivo_occ[nivo_occ[var] != "mq"].shape[0] )/interval_days
            
        #nivo_occ['previous_date'] = nivo_occ['date'].shift(1)
        #nivo_occ['previous_date'] = nivo_occ['previous_date'].fillna(min(nivo_occ['date']))
        #nivo_occ['diff_days'] = nivo_occ.apply(lambda x: x['date'] - x['previous_date'], axis = 1)
        
        #    completude_station_periode = completude_station_periode.\
             #        append(pandas.DataFrame(data = {'station' : [sta], 'periode' : [y], 'count': count_occ}))
  
    #enregistrement excel ?              
    if (save_excel == True):
        completude_station_periode_full.to_excel('data/output/completude_periodes.xlsx')
        
    return(completude_station_periode_full)
        
                











def delete_bad_lines_and_to_batch(table_stations_periods_filter, nivo, 
                                  mois_min, jour_min, mois_max, jour_max):

    #liste des stations reportees
    liste_stations_reporte = table_stations_periods_filter['station'].to_list()
    liste_periodes_reporte = table_stations_periods_filter['periode'].to_list()


    #table vide
    nivo_complete_morning_cleaned = pandas.DataFrame()

    #pour chaque station et sur une periode donnee (15dec-15avr d'une saison)
    for k in range(0, table_stations_periods_filter.shape[0]):
    
        #filtre sur la station occurente
        nivo_tampon = nivo[nivo['Nom'] == liste_stations_reporte[k]]
    
        #filtre sur la date occurente :
        #attention aux annees bisextiles avec le 29 fevrier, necessitant de filtrer sur un jour en moins ces annees
        if ((liste_periodes_reporte[k] == 1999) or (liste_periodes_reporte[k] == 2003) or (liste_periodes_reporte[k] == 2007) or
                (liste_periodes_reporte[k] == 2011) or (liste_periodes_reporte[k] == 2015) or (liste_periodes_reporte[k] == 2019)):
            #hiver avec un 29 fevrier (annees 2012, 2016, 2020)
            nivo_tampon = nivo_tampon[(nivo_tampon['date'] > datetime.datetime(liste_periodes_reporte[k], 
                                                              mois_min, jour_min).date()) & 
                                      (nivo_tampon['date'] < datetime.datetime(liste_periodes_reporte[k]+1,
                                                              mois_max, jour_max-1).date())]
        else:
            #hiver sans 29 fevrier
            nivo_tampon = nivo_tampon[(nivo_tampon['date'] > datetime.datetime(liste_periodes_reporte[k], 
                                                              mois_min, jour_min).date()) & 
                                      (nivo_tampon['date'] < datetime.datetime(liste_periodes_reporte[k]+1, 
                                                              mois_max, jour_max).date())]
            
    
        #traitement remplacement mq et hauteur_neige en numerique
        nivo_tampon = nivo_tampon.replace('mq', numpy.nan)
        nivo_tampon['hauteur_neige'] = pandas.to_numeric(nivo_tampon['hauteur_neige'])

        #nivo count
        nivo_count_tampon = nivo_tampon.groupby('date').apply(lambda x: x['date'].count())
        nivo_count_tampon = nivo_count_tampon.to_frame()
        nivo_count_tampon = nivo_count_tampon.rename(columns = {0: 'count'})
        nivo_count_tampon['date'] = nivo_count_tampon.index
        nivo_count_tampon = nivo_count_tampon.reset_index(drop = True)

        #merge du count. On verra ainsi lorsqu'il y a 2 comptages par jour
        nivo_tampon = nivo_tampon.merge(right = nivo_count_tampon, how = 'inner', on = 'date')



        #on calcule la difference de neige entre 2 lignes consecutives
        #le but etant de filtrer sur les valeurs aberrantes, qui peuvent souvent arriver.
        nivo_tampon['diff'] = numpy.nan

        #pour chacune des lignes de la periode pour la station. On de prend pas les 2premieres et deux dernieres pour des calculs de decalage de lignes
        for i in range(2, nivo_tampon.shape[0] - 2):
    
            #valeurs precedentes, occurentes, et suivantes des hauteurs de neige
            ht_i_minus_2 = nivo_tampon.loc[i-2, ['hauteur_neige']].values
            ht_i_minus_1 = nivo_tampon.loc[i-1, ['hauteur_neige']].values
            ht_i = nivo_tampon.loc[i, ['hauteur_neige']].values
            ht_i_plus_1 = nivo_tampon.loc[i+1, ['hauteur_neige']].values
            ht_i_plus_2 = nivo_tampon.loc[i+2, ['hauteur_neige']].values
    
            #si une seule ligne
            if(nivo_tampon.loc[i, ['count']].values == 1):
        
                #idem precedente et idem suivante
                if((nivo_tampon.loc[i-1, ['count']].values == 1)&(nivo_tampon.loc[i+1, ['count']].values == 1)):
                    nivo_tampon.loc[i, ['diff']] = abs((ht_i_minus_2 + ht_i_minus_1 + ht_i_plus_1 + ht_i_plus_2)/4 - ht_i )
    
                #idem precedente mais pas suivante
                elif((nivo_tampon.loc[i-1, ['count']].values == 1)&(nivo_tampon.loc[i+1, ['count']].values != 1)):
                    nivo_tampon.loc[i, ['diff']] = abs((ht_i_minus_2 + ht_i_minus_1)/2 - ht_i)
    
                #idem suivante mais pas precente
                elif((nivo_tampon.loc[i-1, ['count']].values != 1)&(nivo_tampon.loc[i+1, ['count']].values == 1)):
                    nivo_tampon.loc[i, ['diff']] = abs((ht_i_plus_2 + ht_i_plus_1)/2 - ht_i)
    
            #si deux lignes
            elif(nivo_tampon.loc[i, ['count']].values == 2):
        
                #premiere des deux lignes
                if((nivo_tampon.loc[i-1, ['count']].values != 2)&(nivo_tampon.loc[i+1, ['count']].values == 2)):
                    nivo_tampon.loc[i, ['diff']] = abs((ht_i_minus_2 + ht_i_minus_1 +  ht_i_plus_2)/3 - ht_i )
        
                #seconde des deux lignes
                elif((nivo_tampon.loc[i-1, ['count']].values != 2)&(nivo_tampon.loc[i+1, ['count']].values == 1)):
                    nivo_tampon.loc[i, ['diff']] = abs((ht_i_minus_2 + ht_i_plus_1 +  ht_i_plus_2)/3 - ht_i )
        
            else:
                nivo_tampon.loc[i, ['diff']] = 0
    
    
    
        #supression de lignes : divers cas 
    
        # 1) suppression des lignes avec count = 3 ou plus : rare et trop complique a gerer
        nivo_tampon = nivo_tampon[nivo_tampon['count'] < 3]
    
        # 2) cas ou 2 lignes, il n'y a pas "hauteur_neige" dans l'un des deux -> on supprime la ligne sans hauteur_neige
        nivo_tampon = nivo_tampon[~((nivo_tampon['count'] == 2) & (nivo_tampon['hauteur_neige'].isna() ))]
        nivo_tampon = nivo_tampon.reset_index(drop=True)

        # 3) cas 2 lignes, ou il manque beaucoup d'autres valeurs (temperature, humidite..) dans l'un des deux -> on supprime cette ligne
        nivo_tampon['count_na'] = 0
        for i in range(0, nivo_tampon.shape[0]):
            nivo_tampon.loc[i, ['count_na']] = sum(nivo_tampon.loc[i,].isna() )
            nivo_tampon = nivo_tampon[~((nivo_tampon['count'] == 2) & (nivo_tampon['count_na'] > 4 ))]

    
        # 4) cas ou 2 lignes, le diff est trop élevé
        #a) quand c'est deux lignes : on supprime la ligne avec la grande diff
        nivo_tampon = nivo_tampon[~((nivo_tampon['count'] == 2) & (nivo_tampon['diff'] > 0.4 ))]
        

        #b) quand c'est une ligne : on remplit le 'hauteur_neige' concerne par NA
        nivo_tampon.loc[((nivo_tampon['count'] == 1) & (nivo_tampon['diff'] > 0.4 )), ['hauteur_neige'] ] = numpy.nan

        nivo_tampon = nivo_tampon.reset_index(drop=True)
        
        # 5) cas ou 2 lignes mais l'une n'a pas passé les filtres (les deux peuvent etre bonnes)
        #on garde alors celle avec le diff le plus petit
        nivo_tampon = nivo_tampon.sort_values(by = ['date', 'diff']).groupby('date', as_index = False).first()
    
        nivo_tampon = nivo_tampon.drop(['hour', 'count', 'diff', 'count_na'], axis = 1)
    
        #nom batch
        nivo_tampon['batch'] = liste_stations_reporte[k] + "_"+ str(liste_periodes_reporte[k])

        #ajout a la table
        nivo_complete_morning_cleaned = nivo_complete_morning_cleaned.append(nivo_tampon)
    
        print(liste_stations_reporte[k], liste_periodes_reporte[k], ' ok')
    
    #return
    return(nivo_complete_morning_cleaned)

    
    
    
    
    
    
def creation_table_all_stations_all_dates(date_min_interval, date_max_interval, liste_stations):

    #table qui va venir se joindre a la fin de la grande table, avec la date occurente dans la boucle
    liste_stations_current_date = liste_stations.copy()
    liste_stations['date'] = date_min_interval

    #creation de la boucle : on cree une table avec toutes les stations pour toutes les dates de l'intervalle 
    date_current = date_min_interval + datetime.timedelta(days = 1)
    while date_current < date_max_interval:
        liste_stations_current_date['date'] = date_current
        liste_stations = liste_stations.append(other = liste_stations_current_date, ignore_index = True)
        date_current = date_current + datetime.timedelta(days = 1)

    return(liste_stations)
    
    
    
def to_batch_tables(table, table_batch, mois_min, jour_min, mois_max, jour_max):
    #filtre uniquement sur les dates des batch
    nivo_complete_morning_cleaned_batch = pandas.DataFrame()
    
    #liste des stations reportees
    liste_stations_reporte = table_batch['station'].to_list()
    liste_periodes_reporte = table_batch['periode'].to_list()

    for k in range(0, table_batch.shape[0]):
    
        #filtre sur la station occurente
        nivo_tampon = table[table['Nom'] == liste_stations_reporte[k]]
    
        #filtre sur la date occurente. Attention, annees bissextiles, on retire un jour a la fin
        if ((liste_periodes_reporte[k] == 1999) or (liste_periodes_reporte[k] == 2003) or (liste_periodes_reporte[k] == 2007) or 
            (liste_periodes_reporte[k] == 2011) or (liste_periodes_reporte[k] == 2015) or (liste_periodes_reporte[k] == 2019)):
            nivo_tampon = nivo_tampon[(nivo_tampon['date'] > datetime.datetime(liste_periodes_reporte[k], 
                                                         mois_min, jour_min).date()) & 
                                  (nivo_tampon['date'] < datetime.datetime(liste_periodes_reporte[k]+1, 
                                                         mois_max, jour_max-1).date())]
        else:
            nivo_tampon = nivo_tampon[(nivo_tampon['date'] > datetime.datetime(liste_periodes_reporte[k], 
                                                         mois_min, jour_min).date()) & 
                                  (nivo_tampon['date'] < datetime.datetime(liste_periodes_reporte[k]+1, 
                                                         mois_max, jour_max).date())]
    

        nivo_complete_morning_cleaned_batch = nivo_complete_morning_cleaned_batch.append(nivo_tampon)
        
    return(nivo_complete_morning_cleaned_batch)
    
