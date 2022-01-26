# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:08:11 2021

@author: ichid
"""

####### DASH of mountain data

import datetime as dt

import plotly_express as px

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

#features

date_yesterday = dt.date.today()-dt.timedelta(days=1)



app = dash.Dash(__name__)
    

app.layout = html.Div(children=[
            
            html.H1(children = f'Tableau de bord des mesures d\'enneigement dans les montagnes françaises',
                    style={'textAlign': 'center', 'color': '#7FDBFF'}),
            
            #choix date
            dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed = dt.date(1996, 11, 1),
                    max_date_allowed = date_yesterday,
                    initial_visible_month = date_yesterday,
                    date = date_yesterday),
            
            
            #choix station
            dcc.Dropdown(
                    id = 'station-choice',
                    options = [
                            {'label': i, 'value':i} for i in liste_stations.Nom.unique()
                            ], 
                    multi = False,
                    placeholder = 'Station'
                    )
            
            
            ])


if __name__ == '__main__':
    app.run_server(debug=False)