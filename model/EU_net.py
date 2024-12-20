

# %% Import packages

import numpy as np
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates
import os
from model.YUPPY import  Network # import_generated_scenario
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario
#os.chdir("C:/Users/ghjub/codes/MOPTA/model")




#%% EU
def EU(n_scenarios = 5, init_method = 'day_night_aggregation' ):
    """
    Initializes a network object for the European Union.

    This function creates a network object for the European Union by initializing the necessary attributes. It sets the index of the 'EU' DataFrame to 'node' and assigns values to the 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh' columns. It also creates two DataFrames, 'EU_e' and 'EU_h', with columns 'start_node' and 'end_node'. The 'EU_e' and 'EU_h' DataFrames are assigned values to the 'NTC' and 'MH' columns respectively. Finally, it creates a 'costs' DataFrame with columns 'node', 'cs', 'cw', 'ch', 'chte', 'ceth', 'cNTC', and 'cMH'.

    Returns:
        None
    """
    
    max_wind = 4 
    max_solar = 10 / 1000


    EU=pd.DataFrame([['Italy','Italy',41.90,12.48],
                    ['Spain','Spain',40.42,-3.70],
                    ['Austria','Austria',48.21,16.37],
                    ['France','France',48.86,2.35],
                    ['Germany','Germany',51.15,10.45]], 
                    columns=['node','location','lat','long'])
                
    #ho spostato qui così è un parametro di input
    EU['Mhte']=10**10  # maximum hydrogen to electricuty hourly
    EU['Meth']=10**10  # maximum electricity to hydrogen hourly
    EU['feth']=0.8  # efficiency of electricity in hydrogen
    EU['fhte']=0.8  # effficiency of electricity in hydrogen
    EU['Mns'] = 10**12
    EU['Mnw'] = 10**8
    EU['Mnh'] = 10**12
    EU['MP_wind'] = max_wind
    EU['MP_solar'] = max_solar
    EU['meanP_load'] = 1
    EU['meanH_load'] = 1

    EU_e=pd.DataFrame([  ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria'],
                        ['Germany','Austria'],
                        ['Germany','France'] ],columns=['start_node','end_node'])

    EU_h=pd.DataFrame([  ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria'],
                        ['Germany','Austria'],
                         ['Germany','France'] ],columns=['start_node','end_node'])

    #ho spostato qui così è un parametro di input
    EU_e['NTC']=1000  # maximum transportation  for electricity
    EU_h['MH']=500  # maximum transportation for hydrogen

    #costs
    costs = pd.DataFrame([["All",5000, 3000000, 10,0,0.001,0.001,100,10000,4,0.5]],columns=["node","cs", "cw","ch", "ch_t","chte","ceth","cNTC","cMH","cH_edge","cP_edge"])
    eu = Network(init_method = init_method )
    eu.n = EU
    eu.edgesP = EU_e
    eu.edgesH = EU_h
    eu.costs = costs

    #Import scenarios
    #01_scenario_generation\scenarios\electricity_load_2023.csv
    path = "model/scenario_generation/scenarios/"
    elec_load_df = pd.read_csv(path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR','DE']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR','DE']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France','Germany'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    ave= [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229]
    a = xr.DataArray(ave,dims=['node'], coords={'node':['Italy','Spain','Austria','France','Germany']})
    elec_load_scenario=elec_load_scenario*a
   
    scenario = 0
    
    wind_scenario = import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = import_scenario(path + 'small-eu-PV-scenarios.csv')
    hydro = import_generated_scenario(path+'hydrogen_demandg.csv',5, scenario, node_names=['Italy', 'Spain', 'Austria', 'France','Germany'])
    #hydro_mean = hydro.mean(dim = ["time","scenario"])
    hydrogen_demand_scenario = hydro * (a/500)
    
    
    eu.n_scenarios = n_scenarios
    eu.add_scenarios(wind_scenario * max_wind, pv_scenario * max_solar, hydrogen_demand_scenario, elec_load_scenario)
    eu.init_time_partition()

    return eu


#%%

def EU_big(n_scenarios = 5, init_method = 'day_night_aggregation' ):
    """
    Initializes a network object for the European Union.

    This function creates a network object for the European Union by initializing the necessary attributes. It sets the index of the 'EU' DataFrame to 'node' and assigns values to the 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh' columns. It also creates two DataFrames, 'EU_e' and 'EU_h', with columns 'start_node' and 'end_node'. The 'EU_e' and 'EU_h' DataFrames are assigned values to the 'NTC' and 'MH' columns respectively. Finally, it creates a 'costs' DataFrame with columns 'node', 'cs', 'cw', 'ch', 'chte', 'ceth', 'cNTC', and 'cMH'.

    Returns:
        None
    """
    
    max_wind = 4 
    max_solar = 10 / 1000
# to do: lat and long for map
    EU=pd.DataFrame([['Italy','Italy',41.90,12.48],
                    ['Spain','Spain',40.42,-3.70],
                    ['Austria','Austria',48.21,16.37],
                    ['France','France',48.86,2.35],
                    ['Germany','Germany',51.15,10.45],
                    ['Belgium','Belgium',48.86,2.35],
                    ['Bulgaria','Bulgaria',48.86,2.35],
                    ['Czechia','Czechia',48.86,2.35],
                    ['Denmark','Denmark',48.86,2.35],
                    ['Estonia','Estonia',48.86,2.35],
                    ['Ireland','Ireland',48.86,2.35],
                    ['Greece','Greece',48.86,2.35],
                    ['Croatia','Croatia',48.86,2.35],
                    ['Latvia','Latvia',48.86,2.35],
                    ['Lithuania','Lithuania',48.86,2.35],
                    ['Hungary','Hungary',48.86,2.35],
                    ['Netherlands','Netherlands',48.86,2.35],
                    ['Poland','Poland',48.86,2.35],
                    ['Portugal','Portugal',48.86,2.35],
                    ['Romania','Romania',48.86,2.35],
                    ['Slovenia','Slovenia',48.86,2.35],
                    ['Slovakia','Slovakia',48.86,2.35],
                    ['Finland','Finland',48.86,2.35],
                    ['Sweden','Sweden',48.86,2.35],
                    ['UK','UK',48.86,2.35],
                    ['Norway','Norway',48.86,2.35]], 
                    columns=['node','location','lat','long'])
                
    #ho spostato qui così è un parametro di input
    EU['Mhte']=10**10  # maximum hydrogen to electricuty hourly
    EU['Meth']=10**10  # maximum electricity to hydrogen hourly
    EU['feth']=0.8  # efficiency of electricity in hydrogen
    EU['fhte']=0.8  # effficiency of electricity in hydrogen
    EU['Mns'] = 10**12
    EU['Mnw'] = 10**8
    EU['Mnh'] = 10**12
    EU['MP_wind'] = max_wind
    EU['MP_solar'] = max_solar
    EU['meanP_load'] = 1
    EU['meanH_load'] = 1

    EU_e=pd.DataFrame([ ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria'],
                        ['Germany','Austria'],
                        ['Norway','Sweden'],
                        ['Norway','Germany'],
                        ['Norway','UK'],
                        ['Norway','Denmark'],
                        ['Finland','Sweden'],
                        ['Finland','Estonia'],
                        ['Latvia','Estonia'],
                        ['Latvia','Lithuania'],
                        ['Poland','Germany'],
                        ['Poland','Lithuania'],
                        ['Poland','Czechia'],
                        ['Poland','Slovakia'],
                        ['Portugal','Spain'],
                        ['France','UK'],
                        ['Germany','UK'],
                        ['Belgium','UK'],
                        ['Netherlands','UK'],
                        ['Ireland','UK'],
                        ['Germany','Denmark'],
                        ['Germany','Czechia'],
                        ['Germany','Netherlands'],
                        ['Germany','Belgium'],
                        ['France','Belgium'],
                        ['Austria','Czechia'],
                        ['Austria','Slovakia'],
                        ['Austria','Slovenia'],
                        ['Austria','Hungary'],
                        ['Slovakia','Hungary'],
                        ['Slovakia','Czechia'],
                        ['Hungary','Romania'],
                        ['Greece','Italy'],
                        ['Greece','Bulgaria'],
                        ['Bulgaria','Romania'],
                        ['Germany','France'] ],columns=['start_node','end_node'])

    EU_h=pd.DataFrame([ ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria'],
                        ['Germany','Austria'],
                        ['Norway','Sweden'],
                        ['Norway','Germany'],
                        ['Norway','UK'],
                        ['Norway','Denmark'],
                        ['Finland','Sweden'],
                        ['Finland','Estonia'],
                        ['Latvia','Estonia'],
                        ['Latvia','Lithuania'],
                        ['Poland','Germany'],
                        ['Poland','Lithuania'],
                        ['Poland','Czechia'],
                        ['Poland','Slovakia'],
                        ['Portugal','Spain'],
                        ['France','UK'],
                        ['Germany','UK'],
                        ['Belgium','UK'],
                        ['Netherlands','UK'],
                        ['Ireland','UK'],
                        ['Germany','Denmark'],
                        ['Germany','Czechia'],
                        ['Germany','Netherlands'],
                        ['Germany','Belgium'],
                        ['France','Belgium'],
                        ['Austria','Czechia'],
                        ['Austria','Slovakia'],
                        ['Austria','Slovenia'],
                        ['Austria','Hungary'],
                        ['Slovakia','Hungary'],
                        ['Slovakia','Czechia'],
                        ['Hungary','Romania'],
                        ['Greece','Italy'],
                        ['Greece','Bulgaria'],
                        ['Bulgaria','Romania'],
                        ['Germany','France'] ],columns=['start_node','end_node'])

    #ho spostato qui così è un parametro di input
    EU_e['NTC']=1000  # maximum transportation  for electricity
    EU_h['MH']=500  # maximum transportation for hydrogen

    #costs
    costs = pd.DataFrame([["All",5000, 3000000, 10,0,0.001,0.001,100,10000,0.5,4]],columns=["node","cs", "cw","ch", "ch_t","chte","ceth","cNTC","cMH","cH_edge","cP_edge"])
    eu = Network(init_method = init_method )
    eu.n = EU
    eu.edgesP = EU_e
    eu.edgesH = EU_h
    eu.costs = costs

    #Import scenarios
    #01_scenario_generation\scenarios\electricity_load_2023.csv
    path = "model/scenario_generation/scenarios/"
    elec_load_df = pd.read_csv(path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR','DE']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR','DE']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France','Germany', 'Belgium', 'Bulgaria', 'Czechia', 'Denmark', 'Estonia', 'Ireland', 'Greece', 'Croatia', 'Latvia', 'Lithuania', 'Hungary', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovenia', 'Slovakia', 'Finland','Sweden','UK','Norway'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )

    ave= [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229, 9007, 4116, 6954, 3936, 921, 3969, 5465, 2036, 738, 1337,4805,12470,18961,5791,6088,1399,2902,9012,14910,790,15352]
    a = xr.DataArray(ave,dims=['node'], coords={'node':['Italy', 'Spain', 'Austria', 'France','Germany', 'Belgium', 'Bulgaria', 'Czechia', 'Denmark', 'Estonia', 'Ireland', 'Greece', 'Croatia', 'Latvia', 'Lithuania', 'Hungary', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovenia', 'Slovakia', 'Finland','Sweden','UK','Norway']})
    elec_load_scenario=elec_load_scenario*a
   
    scenario = 0
    
    wind_scenario = import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = import_scenario(path + 'small-eu-PV-scenarios.csv')
    hydro = import_generated_scenario(path+'hydrogen_demandg.csv',5, scenario, node_names=['Italy', 'Spain', 'Austria', 'France','Germany'])
    #hydro_mean = hydro.mean(dim = ["time","scenario"])
    hydrogen_demand_scenario = hydro * (a/500)
    
    
    eu.n_scenarios = n_scenarios
    eu.add_scenarios(wind_scenario * max_wind, pv_scenario * max_solar, hydrogen_demand_scenario, elec_load_scenario)
    eu.init_time_partition()

    return eu

# %%
