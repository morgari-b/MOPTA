#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:41:40 2024

@author: frulcino
"""

from entsoe import EntsoePandasClient
import pandas as pd


def load_api_key(filepath):
    with open(filepath, 'r') as file:
        api_key = file.readline().strip()
    return api_key

def select_energy_types(df, energy_type_list):
    columns = [col for col in df.columns if col[1] in energy_type_list]
    return df[columns]
api_key_path = '../APIkeys/entsoe.txt'

api_key = load_api_key(api_key_path)
client = EntsoePandasClient(api_key=api_key)

start = pd.Timestamp('20220101', tz='Europe/Brussels')
end = pd.Timestamp('20220102', tz='Europe/Brussels')
country_code = 'BE'  # Belgium ISO code
energy_type_list = ["Wind Offshore"]
# Fetch generation per plant
generation = client.query_generation_per_plant(country_code, start=start, end=end)



