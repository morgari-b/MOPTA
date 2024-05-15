


import pandas as pd

LOAD_23=pd.read_excel('scenarios/monthly_hourly_load_values_2023.xlsx', usecols=['DateUTC','CountryCode','Value'],index_col='DateUTC')

load=pd.DataFrame()
for country in LOAD_23['CountryCode'].unique():
    load[country]=LOAD_23.loc[LOAD_23['CountryCode']==country]['Value'].groupby('DateUTC').mean().asfreq('h',method='ffill')
    load['AL'].loc[:'2023-12-05 22:00:00']=load['AL'].loc[:'2023-12-05 22:00:00'].div(load['AL'].loc[:'2023-12-05 22:00:00'].mean())
    load['AL'].loc['2023-12-05 23:00:00':]=load['AL'].loc['2023-12-05 23:00:00':].div(load['AL'].loc['2023-12-05 23:00:00':].mean())
    load.div(load.mean())


