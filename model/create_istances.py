

# %% Import packages
import json
import pandapower.networks as pn
import numpy as np
import random
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates
import os
from model.network import  Network # import_generated_scenario
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario
from model.OPT_methods import OPT_agg, OPT_agg2, OPT_time_partition, OPT3
#os.chdir("C:/Users/ghjub/codes/MOPTA/model")

def convert_pandapower_to_yuppy(N, HequalP = True, standard_buses = True):
    """
    Converts a pandapower network into a YUPPY Network object.

    Parameters:
        N (pandapowerNet): The pandapower network to convert.

    Returns:
        Network: The converted YUPPY Network object.
    """

    max_wind = 4 
    max_solar = 10 / 1000
    # Initialize the YUPPY Network object
    yuppy_network = Network()

    # Add nodes (buses) to the YUPPY network
    nodes_data = []
    for _, bus in N.bus.iterrows():
        if 'geo' in N.bus.columns:
            geodata = json.loads(bus.geo)["coordinates"]
        node_data = {
            'node': bus.name,
            'location': bus.vn_kv,  # Voltage level as a placeholder for location
            'lat': geodata[0] if 'geo' in N.bus.columns and pd.notna(geodata[0]) else None,
            'long': geodata[1] if 'geo' in N.bus.columns and pd.notna(geodata[1]) else None,
            'Mhte': 0,  # Placeholder for hydrogen-to-electricity capacity
            'Meth': 0,  # Placeholder for electricity-to-hydrogen capacity
            'feth': 0,  # Placeholder for efficiency (electricity-to-hydrogen)
            'fhte': 0,  # Placeholder for efficiency (hydrogen-to-electricity)
            'Mns': 0,   # Placeholder for solar capacity
            'Mnw': 0,   # Placeholder for wind capacity
            'Mnh': 0,   # Placeholder for hydrogen storage capacity
            'MP_wind': 0,  # Placeholder for wind power
            'MP_solar': 0,  # Placeholder for solar power
            'meanP_load': 0,  # Placeholder for mean electricity load
            'meanH_load': 0   # Placeholder for mean hydrogen load
        }
        nodes_data.append(node_data)
    yuppy_network.n = pd.DataFrame(nodes_data).set_index('node')

    if standard_buses:
        yuppy_network.n['Mhte']=10**10  # maximum hydrogen to electricuty hourly
        yuppy_network.n['Meth']=10**10  # maximum electricity to hydrogen hourly
        yuppy_network.n['feth']=0.8  # efficiency of electricity in hydrogen
        yuppy_network.n['fhte']=0.8  # effficiency of electricity in hydrogen
        yuppy_network.n['Mns'] = 10**12
        yuppy_network.n['Mnw'] = 10**8
        yuppy_network.n['Mnh'] = 10**12
        yuppy_network.n['MP_wind'] = 10**18
        yuppy_network.n['MP_solar'] = 10**18
        yuppy_network.n['meanP_load'] = 1
        yuppy_network.n['meanH_load'] = 1
    # Add power edges (lines) to the YUPPY network
    edgesP_data = []
    for _, line in N.line.iterrows():
        edge_data = {
            'start_node': N.bus.loc[line.from_bus, 'name'],
            'end_node': N.bus.loc[line.to_bus, 'name'],
            'NTC': 1000  # Use line length as a placeholder for capacity
        }
        edgesP_data.append(edge_data)
    yuppy_network.edgesP = pd.DataFrame(edgesP_data)

    # Add transformer edges (if applicable)
    if HequalP:
        # Add hydrogen edges (if applicable)
        edgesH_data = []
        for _, line in N.line.iterrows():
            edge_data = {
                'start_node': N.bus.loc[line.from_bus, 'name'],
                'end_node': N.bus.loc[line.to_bus, 'name'],
                'MH': 1000  # Use line length as a placeholder for capacity
            }
            edgesH_data.append(edge_data)
        yuppy_network.edgesH = pd.DataFrame(edgesH_data)
    else:
        # Add hydrogen edges (if applicable)
        edgesH_data = []
        for _, trafo in N.trafo.iterrows():
            edge_data = {
                'start_node': N.bus.loc[trafo.hv_bus, 'name'],
                'end_node': N.bus.loc[trafo.lv_bus, 'name'],
                'MH': 500  # Transformer nominal power as a placeholder
            }
            edgesH_data.append(edge_data)
        yuppy_network.edgesH = pd.DataFrame(edgesH_data)

    # Add scenarios (time-dependent variables)
    # Placeholder for generation and load scenarios
    time_index = pd.date_range(start='2023-01-01', periods=24, freq='H')  # Example hourly time index
    nodes = yuppy_network.n.index.tolist()

    yuppy_network.genW_t = xr.DataArray(
        np.random.rand(len(time_index), len(nodes)),  # Random data as a placeholder
        dims=['time', 'node'],
        coords={'time': time_index, 'node': nodes}
    )
    yuppy_network.genS_t = xr.DataArray(
        np.random.rand(len(time_index), len(nodes)),  # Random data as a placeholder
        dims=['time', 'node'],
        coords={'time': time_index, 'node': nodes}
    )
    yuppy_network.loadP_t = xr.DataArray(
        np.random.rand(len(time_index), len(nodes)),  # Random data as a placeholder
        dims=['time', 'node'],
        coords={'time': time_index, 'node': nodes}
    )
    yuppy_network.loadH_t = xr.DataArray(
        np.zeros((len(time_index), len(nodes))),  # Placeholder for hydrogen demand
        dims=['time', 'node'],
        coords={'time': time_index, 'node': nodes}
    )

    # Add costs (if applicable)
    costs = pd.DataFrame([["All",5000, 3000000, 10,0.01,0.001,0.001,100,1000,4,1]],columns=["node","cs", "cw","ch", "ch_t","chte","ceth","cNTC","cMH","cH_edge","cP_edge"])
    yuppy_network.costs =costs

    # Set the number of scenarios and time steps
    yuppy_network.n_scenarios = 1  # Default to 1 scenario
    yuppy_network.d = 1  # Default to 1 scenario dimension
   
    # add scenarios
    #Import scenarios
    #01_scenario_generation\scenarios\electricity_load_2023.csv
    
    #now I want to copy them so that 
    return yuppy_network

def assign_random_scenarios_to_nodes(network, n_scenarios = 2):
    """
    Randomly assigns a scenario (Italy, Spain, Austria, France, Germany) to each node in the given Network object.

    Parameters:
        network (Network): The Network object to which scenarios will be assigned.

    Returns:
        Network: The updated Network object with scenarios assigned to each node.
    """
    # Define the path to the scenario files
    path = "model/scenario_generation/scenarios/"
    
    # Load electricity load data
    elec_load_df = pd.read_csv(path + 'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR', 'DE']]
    time_index = range(elec_load_df.shape[0])  # Time index for the scenarios

    # Create an xarray DataArray for electricity load scenarios
    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR', 'DE']].values, axis=2),  # Add one dimension for scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France', 'Germany'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )

    # Scaling factors for each country
    ave = [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229]
    scaling_factors = xr.DataArray(ave, dims=['node'], coords={'node': ['Italy', 'Spain', 'Austria', 'France', 'Germany']}) / 100
    print("random division by 10 to diminish the load.")
    elec_load_scenario = elec_load_scenario * scaling_factors

    # Load wind and PV scenarios
    wind_scenario = import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = import_scenario(path + 'small-eu-PV-scenarios.csv')

    # Load hydrogen demand scenarios
    hydro = import_generated_scenario(path + 'hydrogen_demandg.csv', 5, 0, node_names=['Italy', 'Spain', 'Austria', 'France', 'Germany'])
    hydrogen_demand_scenario = hydro * (scaling_factors / 500)

    # # List of available countries/scenarios
    countries = ['Italy', 'Spain', 'Austria', 'France', 'Germany']

    # Randomly assign a scenario to each node in the network
    assigned_scenarios = {}
    for node in network.n.index:
        assigned_country = random.choice(countries)
        assigned_scenarios[node] = assigned_country

    #Create new DataArrays for the network based on the assigned scenarios
    genW_t = []
    genS_t = []
    loadP_t = []
    loadH_t = []
   
    for node in network.n.index:
        country = assigned_scenarios[node]
        country_index = countries.index(country)

        # Extract the scenario for the assigned country
        genW_t.append(wind_scenario.sel(node=country))
        genS_t.append(pv_scenario.sel(node=country))
        loadP_t.append(elec_load_scenario.sel(node=country))
        loadH_t.append(hydrogen_demand_scenario.sel(node=country))

    # Combine the scenarios into a single DataArray for the network
    genW_t = xr.concat(genW_t, dim='node').assign_coords(node=network.n.index).transpose('time','node','scenario')
    genS_t = xr.concat(genS_t, dim='node').assign_coords(node=network.n.index).transpose('time','node','scenario')
    loadP_t = xr.concat(loadP_t, dim='node').assign_coords(node=network.n.index).transpose('time','node','scenario')
    loadH_t = xr.concat(loadH_t, dim='node').assign_coords(node=network.n.index).transpose('time','node','scenario')
    network.n_scenarios = n_scenarios
    network.T = genW_t.time.shape[0]
    network.add_scenarios(genW_t,genS_t,loadP_t,loadH_t)
    network.init_time_partition()
    return network


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
    costs = pd.DataFrame([["All",5000, 3000000, 10,0,0.001,0.001,100,10000,100,100]],columns=["node","cs", "cw","ch", "ch_t","chte","ceth","cNTC","cMH","cH_edge","cP_edge"])
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
if __name__ == "__main__":
       # Create a sample Network object
    pandan =  pn.case9()
    n = convert_pandapower_to_yuppy(pandan)
    n = assign_random_scenarios_to_nodes(n)
    n.init_time_partition()
    print(n.T)
    #%%
    eu = EU(n_scenarios=1, init_method = 'day_night_aggregation')
    #%%
    eu_res = OPT_agg(eu)
    #%%
    eu_res1 = OPT_agg2(eu, N_iter = 1, iter_method = 'total')

    #%%
    #eu_res2 = OPT_time_partition(eu, N_iter = 10)
    #%%
    eu_res3 = OPT3(eu)
    #%%
    iterated_eu = eu_res1[-1]['network']
    eu_res4 = OPT_agg(iterated_eu)

    #%%
    res = OPT_agg(n)
    # %%
    res1 = OPT_agg_correct(n)
    #%%
    res2 = OPT_agg2(n, N_iter = 10)

    # %%
