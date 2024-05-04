#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:41:40 2024

@author: frulcino
"""

# %% import
from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
# %% define functions
def load_api_key(filepath):
    with open(filepath, 'r') as file:
        api_key = file.readline().strip()
    return api_key

def select_energy_types(df, energy_type_list):
    columns = [col for col in df.columns if col[1] in energy_type_list]
    return df[columns]


def fetch_data(start, end, country_code, api_key, max_retries = 3):
    client = EntsoePandasClient(api_key=api_key)
    # Set the start and end date
    start_date = pd.Timestamp(f'{start}', tz='Europe/Brussels')
    end_date = pd.Timestamp(f'{end}', tz='Europe/Brussels')
    
    #def fetch_data_instance(country_code, start=start_date, end=current_end_date):
    # Create an empty DataFrame to store the aggregated data
    full_generation_data = pd.DataFrame()

    # Loop over the date range in steps of 5 days
    while start_date < end_date:
        # Calculate the end of the current 5-day period
        current_end_date = start_date + timedelta(days=4)
        if current_end_date > end_date:
            current_end_date = end_date

        try:
            # Fetch generation data for the current period
            print(f"Fetching data from {start_date} to {current_end_date}")
            generation = client.query_generation_per_plant(country_code, start=start_date, end=current_end_date)
            full_generation_data = pd.concat([full_generation_data, generation])
        except Exception as e:
            print(f"Error fetching datam retrying: {e}")
            for i in np.arange(max_retries):
                try:
                    # Fetch generation data for the current period
                    print(f"Fetching data from {start_date} to {current_end_date}")
                    generation = client.query_generation_per_plant(country_code, start=start_date, end=current_end_date)
                    full_generation_data = pd.concat([full_generation_data, generation])
                    break
                except Exception as e:
                    print(f"Error fetching data for {i+2} time: {e}")
        finally:
            # Increment start_date to the next interval
            start_date = current_end_date + timedelta(days=1)

    return full_generation_data


# The script now tries to fetch the data, and no matter the outcome, it attempts to save what has been fetched.


# %% load key
api_key_path = '../APIkeys/entsoe.txt'
api_key = load_api_key(api_key_path)
client = EntsoePandasClient(api_key=api_key)


# %% set parameters 
start = pd.Timestamp('20220301', tz='Europe/Brussels')
end = pd.Timestamp('20220306', tz='Europe/Brussels')
country_code = 'BE'  # Belgium ISO code
energy_type_list = ["Wind Offshore", "Wind Onshore"]
average_wind_generation = 4*2.539914194598637 #hour wind generation for 1h intervals
# %% fetch data
# Fetch generation per plant
#generation = client.query_generation_per_plant(country_code, start=start, end=end)

try:
    generation_data = fetch_data(start, end, country_code, api_key)
finally:
    # Save the data to CSV regardless of whether the fetch was completely successful
    filename = f'generation_data_{country_code}_{2022}_to_{2023}.csv'
    generation_data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

#generation = select_energy_types(generation, energy_type_list)
# %% data manipulation

#generation.columns = [col[0].split(" ")[0] + col[0].split(" ")[-1] for col in generation.columns] #rename columns
#n_gens = (generation.mean()/average_wind_generation).astype(int) #get n_gens for each plant
#generation = generation / n_gens #transform to per unit MW production
# %%
#generation.plot()

# %% save data