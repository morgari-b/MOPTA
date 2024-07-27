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
from YUPPY import OPT1, OPT2, Network, import_generated_scenario
#os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")



#%% EU
def EU():
    """
    Initializes a network object for the European Union.

    This function creates a network object for the European Union by initializing the necessary attributes. It sets the index of the 'EU' DataFrame to 'node' and assigns values to the 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh' columns. It also creates two DataFrames, 'EU_e' and 'EU_h', with columns 'start_node' and 'end_node'. The 'EU_e' and 'EU_h' DataFrames are assigned values to the 'NTC' and 'MH' columns respectively. Finally, it creates a 'costs' DataFrame with columns 'node', 'cs', 'cw', 'ch', 'chte', 'ceth', 'cNTC', and 'cMH'.

    Returns:
        None
    """
 
    EU=pd.DataFrame([['Italy',41.90,12.48],
                    ['Spain',40.42,-3.70],
                    ['Austria',48.21,16.37],
                    ['France',48.86,2.35]], 
                    columns=['node','lat','long']).set_index('node')
                
    #ho spostato qui così è un parametro di input
    EU['Mhte']=10**6  # maximum hydrogen transport cost
    EU['Meth']=10**5  # maximum electricity transport cost
    EU['feth']=0.7  # fraction of electricity in hydrogen
    EU['fhte']=0.75  # fraction of electricity in hydrogen
    EU['Mns'] = 10000
    EU['Mnw'] = 10000
    EU['Mnh'] = 10000

    EU_e=pd.DataFrame([  ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria']  ],columns=['start_node','end_node'])

    EU_h=pd.DataFrame([  ['France','Italy'],
                        ['Austria','Italy'],
                        ['France','Spain'],
                        ['France','Austria']  ],columns=['start_node','end_node'])

    #ho spostato qui così è un parametro di input
    EU_e['NTC']=1000  # maximum transportation cost for electricity
    EU_h['MH']=500  # maximum transportation cost for hydrogen

    #costs
    costs = pd.DataFrame([["All",4000, 3000000, 10,2,200,1000,10000]],columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])
    eu = Network()
    eu.n = EU
    eu.edgesP = EU_e
    eu.edgesH = EU_h
    eu.costs = costs

    # Import scenarios
    #01_scenario_generation\scenarios\electricity_load_2023.csv
    scen_path = '../01_scenario_generation/scenarios/'
    elec_load_df = pd.read_csv(scen_path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    
    scenario = 0
    wind_scenario = import_generated_scenario(scen_path+'wind_scenarios.csv',4,scenario, node_names= ['Italy', 'Spain', 'Austria', 'France'])
    pv_scenario = import_generated_scenario(scen_path+'PV_scenario.csv',4, scenario, node_names=['Italy', 'Spain', 'Austria', 'France'])
    hydrogen_demand_scenario = import_generated_scenario(scen_path+'hydrogen_demandg.csv',4, scenario, node_names=['Italy', 'Spain', 'Austria', 'France'])


    eu.add_scenarios(wind_scenario, pv_scenario, hydrogen_demand_scenario, elec_load_scenario)
    return eu

