# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 18:43:57 2021

@author: ichid
"""

import os
import datetime as dt
import requests

#fonction d'extraction automatique du .csv du serveur de Meteo-France, si c'est demande
#il n'y a pas de fonction en entree. Il faut juste la date du jour
#les donnees dispos s'arretant la veille, on doit adapter l'extraction du fichier en fonction 
#(ex : si on est le 1er mars 2021, on va extraire fevrier 2021
def extract_last_data():
    
    #liste des fichiers deja dispo
    list_nivo = os.listdir('data/input/nivo')
    
    
    ###EXTRACTION DERNIER MOIS
    #si premier du mois : on doit extraire le mois precedent
    if (dt.date.today().day == 1):
        
        #si 1er janvier - il faut extraire decembre de l'annee precedente
        if (dt.date.today().month == 1):
            this_year = str(dt.date.today().year - 1)
            this_month = '12'
            
        #si 1er du mois autre que janvier (on ajoute zero pour les mois a un chiffre, i.e. jusqu'a septembre)
        else: 
            this_year = str(dt.date.today().year)
            if (len(str(dt.date.today().month)) == 1):
                this_month = '0'+str(dt.date.today().month)
            else:
                this_month = str(dt.date.today().month)


    #si autre jour que 1er du mois : on peut extraire ce mois-la !
    else:
        this_year = str(dt.date.today().year)
        if (len(str(dt.date.today().month)) == 1):
            this_month = '0'+str(dt.date.today().month)
        else:
            this_month = str(dt.date.today().month)

    #url ou l'on va venir extraire les donnees : on vient mettre a la fin de l'URL le mois et l'annee
    url = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Nivo/Archive/nivo."+ this_year + this_month + ".csv.gz"

    #enregistrement du fichier extrait, avec le nom 'nivo.annee.mois.csv'
    filename = "data/input/nivo/nivo." + this_year + this_month + ".csv"
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    print('fichier extrait : nivo.' + this_year + this_month + '.csv' )
        
        
        
    ###EXTRACTION DE MOIS INTERMEDIAIRES (si il y en a)
    #par exemple, dernier dans le dossier est 2021/04, on vient ci-dessus d'extraire 2021/06, 
    #il va falloir extraire 2021/05, mais aussi 2021/04 qui n'est probablement pas complet car extrait avant le 30/31
    
    #dernier fichier dispo (extrait donc dans le passé)
    last_nivo_file = list_nivo[len(list_nivo)-1 ]
    
    #constitution liste des fichiers / mois intermediaires. Attention, on va remettre le dernier mois dispo a jour aussi
    list_nivo_theoric = []
    for y in range(1996, int(this_year)+1):
        for m in range(1,13):
            y_str = str(y)
            if (len(str(m)) == 1):
                m_str = '0'+str(m)
            else:
                m_str = str(m)
        
            list_nivo_theoric.append(int(y_str + m_str))
    
    list_nivo_to_extract = [x for x in list_nivo_theoric if (x >= int(last_nivo_file[5:11]))]
    list_nivo_to_extract = [x for x in list_nivo_to_extract if (x < int(this_year + this_month))]
    
    #si il n'y a rien dans la liste, on passe
    if len(list_nivo_to_extract) == 0:
        pass
    
    #si au moins un element :
    else:
        for ym in list_nivo_to_extract:
            #url ou l'on va venir extraire les donnees : on vient mettre a la fin de l'URL le mois et l'annee
            url = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Nivo/Archive/nivo."+ str(ym) + ".csv.gz"

            #enregistrement du fichier extrait, avec le nom 'nivo.annee.mois.csv'
            filename = "data/input/nivo/nivo." + str(ym) + ".csv"
            with open(filename, "wb") as f:
                r = requests.get(url)
                f.write(r.content)
            print('fichier extrait : nivo.' + str(ym) + '.csv' )
    
    
    
    