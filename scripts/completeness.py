# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:57:46 2021

@author: ichid
"""

import matplotlib.pyplot as plt
write_excel_cleaning = False

nivo['month'] = nivo['datetime'].apply(lambda x: dt.date(x.year, x.month, 1) )

month_completeness = nivo.groupby(['month']).count()
month_completeness['month'] = month_completeness.index.astype(str)


plt.figure(figsize=(30,60))
plt.barh(y = month_completeness['month'], width = month_completeness['ht_neige'])
plt.savefig('figures/completude/completude_mois.png')
plt.show()


#ht neige
month_completeness_htneige = nivo[nivo.ht_neige != 'mq'].groupby(['month']).count()
month_completeness_htneige['month'] = month_completeness_htneige.index.astype(str)
month_completeness_htneige

plt.figure(figsize=(30,60))
plt.barh(y = month_completeness_htneige['month'], width = month_completeness_htneige['ht_neige'])
plt.savefig('figures/completude/completude_mois_ht_neige.png')
plt.show()



#altitude des stations

plt.figure(figsize=(10,10))
plt.hist(liste_stations['Altitude'], bins = 20)
plt.savefig('figures/completude/altitude_stations.png')
plt.show()



#table completude par jour (i.e. par jour, combien de stations ont des donnees neige ?)
completude_day = nivo_complete_morning[nivo_complete_morning.hauteur_neige != 'mq'].groupby('date').count()
completude_day['percentage_hauteur_neige'] = completude_day['hauteur_neige']/len(liste_stations.ID.unique())
completude_day = completude_day[['hauteur_neige', 'percentage_hauteur_neige']]

if (write_excel_cleaning == True):
    #on enregistre la table des completude
    completude_day.to_excel('data/output/completude_day.xlsx')






