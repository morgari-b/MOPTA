#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:05:49 2024v
@author: morgari
CURRENT WORKING MODELS:
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates
import os
#os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")
import xarray as xr
import folium
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario, scenario_to_array



#%% import_generated_scenario

def import_scenario_val(start,stop):
    
    path = "model/scenario_generation/scenarios/"
    elec_load_df = pd.read_csv(path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR','DE']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR','DE']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France','Germany'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    
    ave = [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229 ]
    a = xr.DataArray(ave,dims=['node'], coords={'node':['Italy','Spain','Austria','France','Germany']})
    elec_load_scenario=elec_load_scenario*a
    
    #wind_scenario = 4*scenario_to_array(pd.read_csv(path +'small-eu-wind-scenarios3.csv', index_col = 0))
    wind_scenario = 4*import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = 0.01*import_scenario(path + 'small-eu-PV-scenarios.csv')
    
    
    df = pd.read_csv(path+'hydrogen_demandg.csv', index_col = 0).head()
    time_index = range(df.shape[1])#pd.date_range('2023-01-01 00:00:00', periods=df.shape[1], freq='H')
    node_names=['Italy', 'Spain', 'Austria', 'France','Germany']

    hydro = xr.DataArray(
        np.expand_dims(df.T.values, axis = 2),
        coords={'time': time_index, 'node': node_names, 'scenario': [0]},
        dims=['time', 'node', 'scenario'] )
    hydro_mean = hydro.mean(dim = ["time","scenario"]) 
    hydrogen_demand_scenario = hydro / hydro_mean
    
    
    #hydrogen_demand_scenario2 = import_scenario(path + 'hydrogen_demandg.csv')
    
    #eu.add_scenarios(wind_scenario * max_wind, pv_scenario * max_solar, hydrogen_demand_scenario, elec_load_scenario)
    
    scenarios = {
        'wind_scenario' : wind_scenario.sel(scenario=slice(start,stop)),
        'pv_scenario' : pv_scenario.sel(scenario=slice(start,stop)),
        'hydrogen_demand_scenario' : hydrogen_demand_scenario,
        'elec_load_scenario' : elec_load_scenario
        }
    
    
    return scenarios


# %%
#LESS IMPORTANT:
# give name to constriants
#todo: da spostrare in uno script più sensato
#todo: import classic example cases
#todo: some time aggregation plotting

def solution_to_xarray(var, dims, coords):
    values = np.zeros(tuple([len(c) for c in coords]))
    for key in var.keys():
        values[key] = var[key].X
    
    array = xr.DataArray(values,
                        dims=dims,
                        coords = dict(zip(dims,coords)))
    return array



#%% class time_partition

class time_partition:

    def aggregate(self,l, i0,i1):
        """
        Aggregates a list into sublists based on the given indices.

        Args:
            l (list): The list to be aggregated.
            i0 (int): The starting index of the sublist.
            i1 (int): The ending index of the sublist.

        Returns:
            list: The aggregated list with sublists.
        """
        if i0 == 0:
            return [l[i0:i1]]+ l[i1:]
        elif i1 == len(l):
            return l[0:i0]+[l[i0:i1]]
        elif i0 == i1:
            return l
        else:
            return l[0:i0]+[l[i0:i1]]+ l[i1:]
    def disaggregate(self,l,i):
        if type(l[i]) is list:
            return l[0:i] + l[i] + l[i+1:]
        else:
            raise ValueError(f"{l[i]} is not a list and cannot be disaggregated")

    def day_aggregation(self):
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        for day in range(n_days): #ragruppo i giorni
            l = self.aggregate(l, day, day + 24)

        n_seasons = 16
        season = int(np.floor(n_days / n_seasons))  #mettiamo 10 giorni interi per intero
        for i in range(n_seasons):
            l = self.disaggregate(l, i * season + 23*i)
        return l

    def day_night_aggregation(self):
        """
        Aggregates the time steps into sublists based on day and night periods.

        Returns:
            list: The aggregated list with sublists representing day and night periods.
        """
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        l = self.aggregate(l, 0, 6) #initial night
        for day in range(n_days-1): #ragruppo i giorni
            l = self.aggregate(l, 2*day + 1, 2*day + 1 + 14)
            l = self.aggregate(l, 2*day + 2, 2*day + 2 + 10) 
        day = n_days-1
        l = self.aggregate(l, 2*day + 1, 3*day + 1 + 14)
        l = self.aggregate(l, 2*day + 2, 3*day + 2 + 4)
        l1 = []
        for i in l:
            if type(i) is int: #ho modificato qui protrebbe creare problemi, stavo cercando di togliere liste vuote.
                l1 += i
            elif len(i) > 1:
                l1.append(i)
        l = l1
        #season = int(np.floor(n_days /10))  #mettiamo 10 giorni interi per intero
        #for i in range(10):
        #    l = self.disaggregate(l, i * season + 23*i)
        return l

    def __init__(self,T, init_method='day_aggregation'):
        self.T=T
        self.time_steps = list(range(T))
        print(init_method)
        #define initial_aggregation with the chosen method.
        method = getattr(self, init_method)
        if method is not None and callable(method):
            self.agg = method()
        else:
            print(f"Method {init_method} not found or not callable")
        
        self.old_agg = []
        self.family_tree = [] #list of lists of intervals splitted in some way (todo? become a dictionary mapping which interval in splitted into which intervals)

    def len(self,i):
        if type(self.agg[i]) is list:
            return len(self.agg[i])
        else:
            return 1

    def random_iter_partition(self,k=1):
        self.old_agg += [self.agg.copy()]
        family_list = []
        for _ in range(k):
            tp = self.agg
            agg_indices = [i for i in range(len(tp)) if type(tp[i]) is list] #get index of aggregate intervals
            rand_ind = agg_indices[np.random.randint(len(agg_indices))] #get random index
            new_int =self.agg[rand_ind]
            self.agg = self.disaggregate(self.agg,rand_ind)
            family_list += [new_int]
        self.family_tree += [time_partition.order_intervals(family_list)]

    def iter_partition(self, t):
        self.old_agg += [self.agg.copy()]
        family_list = []
        tp = self.agg
        new_int =self.agg[t]
        if new_int is list:
            self.agg = self.disaggregate(self.agg,t)
            family_list += [new_int]
            self.family_tree += [time_partition.order_intervals(family_list)]
        else:
            print(f"{new_int} is a singleton")
    
    
    def iter_partition_intervals(tp_obj, intervals):
        "completly disaggregates intervals in intervals"
        tp_obj.old_agg += [tp_obj.agg.copy()]
        family_list = []
        for t in intervals:
            new_int =tp_obj.agg[t]
            if new_int is list:
                tp_obj.agg = tp_obj.disaggregate(tp_obj.agg,rand_ind)
                family_list += [new_int]
            else:
                print(f"{new_int} is a singleton")
        tp_obj.family_tree += [time_partitiotp_obj.order_intervals(family_list)]

    def to_dict(self):
        """
        Save the Network object as a dictionary of dictionaries, where each attribute is stored as a dictionary.

        Returns:
            dict: A dictionary containing all the attributes of the Network object.
        """
        output = {}
        for attr in self.__dict__:
            attribute = getattr(self, attr)
            if type(attribute) is int:
                output[attr] = attribute
            elif type(attribute) is float:
                output[attr] = attribute
            elif type(attribute) is str:
                output[attr] = attribute
            elif type(attribute) is list:
                output[attr] = attribute
            else:
                output[attr] = getattr(self, attr).to_dict()
        return output

    @staticmethod
    def from_dict(d):
        tp = time_partition(d['T'])
        for attr, value in d.items():
            setattr(tp, attr, value)
        return tp

    @staticmethod
    def tuplize(tp):
        l = []
        for i in tp:
            if type(i) is list:
                l.append(tuple(i))
            else:
                l.append(i)
        return l

    @staticmethod
    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

    @staticmethod
    def interval_subsets(interval,tp):
        """
        Get the indexes of elements in the given time partition `tp` that are subsets of the given `interval`.

        Parameters:
            interval (set): The set of elements to check for subset membership.
            tp (list): The list of elements to search for subsets of `interval`.

        Returns:
            list: The indexes of elements in `tp` that are subsets of `interval`.
        """
        indexes = []
        for i in range(len(tp)):
            l = tp[i]
            if type(l) is list:
                if set(l).issubset(interval):
                    indexes.append(i)
            else:
                if l in interval:
                    indexes.append(i)
        return indexes
        
# %% df_aggregator

def df_aggregator(df, time_partition):
    """
    Aggregates the data in the given DataFrame `df` based on the time aggregation specified in `agg_time`.
    In particular it sums the coordinatees found in the same time interbals in time_partition together
    Parameters:
        df (xarray.DataArray): The DataFrame containing the data to be aggregated.
        time_partition (list): A list of time aggregation specifications. Each element in the list can be either a single time value or a list of time values.
            
    """
    df2 = df.assign_coords({'time': range(df.shape[0])})
    summed_df = []
    for t in time_partition:
        if type(t) is list:
            summed_df.append(df2.sel(time = t).sum(dim='time'))
        else:
            summed_df.append(df2.sel(time = t).drop_vars('time', errors='ignore'))

    add_df = xr.concat(summed_df, dim = 'time', coords = 'minimal', compat = 'override').assign_coords(time = ('time', range(len(time_partition))))

    return add_df


def df_aggregator2(network, df0, prev_df, splitted_intervals, son_indeces_lists):
    """
    Aggregates the data in the given DataFrame `df` based on the time aggregation specified in `network.time_partition`.
    
    Parameters:
        network (Network): The network object containing the time partition specification.
        df0 (xarray.DataArray): The original DataFrame containing the data to be aggregated.
        prev_df (xarray.DataArray): The previous aggregation of df0, respect to the partition old_tp.
        splitted_intervals (list): A list of time intervals that were split from the original time partition.
        son_indeces_lists (list): A list of lists containing the indices of the son intervals in the time partition.
    
    Returns:
        xarray.DataArray: The aggregated DataFrame.
    """
    df00 = df0.assign_coords({'time': range(df0.shape[0])})
    tp_obj = network.time_partition
    old_tp = tp_obj.old_agg[-1]
    tp = tp_obj.agg
    father_indices = [old_tp.index(i) for i in splitted_intervals if i in old_tp]

    slices = []
    
    # Add initial slice
    if father_indices[0] > 0:
        slices.append(prev_df.sel(time=slice(None, father_indices[0]-1)).drop_vars('time', errors='ignore'))
    
    for i in range(len(father_indices) - 1):
        son_indices = son_indeces_lists[i]
        
        # Add son indices slices
        for t in (tp[j] for j in son_indices):
            if isinstance(t, list):
                slices.append(df00.sel(time=t).sum(dim='time').drop_vars('time', errors='ignore'))
            else:
                slices.append(df00.sel(time=t).drop_vars('time', errors='ignore'))
        
        # Add slice between current and next father index
        if father_indices[i] + 1 <= father_indices[i + 1] - 1:
            slices.append(prev_df.sel(time=slice(father_indices[i] + 1, father_indices[i + 1] - 1)).drop_vars('time', errors='ignore'))
    
    # Add final son indices slices
    son_indices = son_indeces_lists[-1]
    for t in (tp[i] for i in son_indices):
        if isinstance(t, list):
            slices.append(df00.sel(time=t).sum(dim='time').drop_vars('time', errors='ignore'))
        else:
            slices.append(df00.sel(time=t).drop_vars('time', errors='ignore'))
    
    # Add final slice
    if father_indices[-1] + 1 < len(prev_df['time']):
        slices.append(prev_df.sel(time=slice(father_indices[-1] + 1, None)).drop_vars('time', errors='ignore'))

    # Concatenate all slices
    new_df = xr.concat(slices, dim='time', coords='minimal', compat='override')

    # Assign new time coordinates
    new_time_coords = range(len(new_df['time']))
    new_df = new_df.assign_coords(time=('time', new_time_coords))
    
    return new_df


def get_rho(n,VARS, n_max = 10):
    """
    given a network and a solution to an aggregated problem, calculates the rho value for each constraint
    """
    #rho for a constraint
    tp = n.time_partition.agg
    I = tp[0]

    P_net = []
    H_net = []
    HL = n.loadH_t
    PL = n.loadP_t
    ES = n.genS_t
    EW = n.genW_t
    nw = xr.DataArray(VARS["nw"], dims='node', coords={'node': HL.coords['node']})
    ns = xr.DataArray(VARS["ns"], dims='node', coords={'node': HL.coords['node']})
    ES = ES.assign_coords(time=PL.coords['time'])
    EW = EW.assign_coords(time=PL.coords['time'])
    Pnett = PL - nw*EW - ns*ES 
    Hnett = HL #we don't haev any hydrogen generations corresponding to time independent variables
    intervals = [i for L in [[k]*len(tp[k]) for k in range(len(tp))] for i in L]
    Pnett = Pnett.assign_coords(interval=('time',intervals))
    Hnett = Hnett.assign_coords(interval=('time',intervals))
        #P_net.append(Pnett)
        #H_net.append(Hnett)

    rhoP = Pnett.groupby('interval') / (Pnett.groupby('interval').sum())
    rhoH = Hnett.groupby('interval') / (Hnett.groupby('interval').sum())
    varhop = rhoP.var(dim='node')
    varhoh = rhoH.var(dim='node')
    varho = varhop + varhoh

    
    return rhoP, rhoH, varho
# 'top_n_coords' now contains the coordinates of the top 'n' values

def xr_top_n(x, n_max, dim = 'time'):
    # Stack all dimensions into a single one for easier sorting
    stacked = x.stack(all_dims=x.dims).copy()
    # Find the coordinates of the top 'n' values
    top_n_coords = {dim: []}
    max_values = []
    for _ in range(n_max):
        # Get the position of the maximum value
        max_idx = stacked.argmax('all_dims')
        max_value = stacked.isel(all_dims=max_idx).values.item()
        max_values.append(max_value)
        # Get the coordinates corresponding to the maximum value
        max_coords = stacked.isel(all_dims=max_idx).coords
        # Store the coordinates
        top_n_coords[dim].append(int(max_coords[dim]))
        # Set the current maximum value to a very low value to exclude it in the next iteration
        stacked[max_idx] = float('-inf')

    return top_n_coords
#%% class Network

class Network:
    """
    Class to represent a network of nodes and edges.
    
    Attributes:
        n (pandas.DataFrame): DataFrame with index name of node, columns lat and long.
        edgesP (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        edgesH (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        loadH (pandas.DataFrame): DataFrame with time dependent variables for hydrogen.
        loadE (pandas.DataFrame): DataFrame with time dependent variables for electricity.
        genW (pandas.DataFrame): DataFrame with time dependent variables for wind.
        genS (pandas.DataFrame): DataFrame with time dependent variables for solar.
    """
    
    def __init__(self, init_method = 'day_night_aggregation' ):
        """
        Initialize a Network object.
        
        Parameters:
            nodes (pandas.DataFrame): DataFrame with index name of node, columns lat and long.
            edgesP (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
            edgesH (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        """
        self.init_method = init_method
        self.n=pd.DataFrame(columns=['node','location','lat','long', 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh','MP_wind','MP_solar','meanP_load','meanH_load']).set_index('node')
        # nodes = pd.DataFrame with index name of node, columns lat and long.
        self.edgesP=pd.DataFrame(columns=['start_node','end_node','NTC'])
        self.edgesH=pd.DataFrame(columns=['start_node','end_node','MH'])
        # edges = pd.DataFrame with columns start node and end node, start coords and end coords.
        self.n_scenarios = 0
        self.costs = pd.DataFrame(columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])

    def add_node(self, new_df ):
        if 'node' in new_df.columns:
            self.n.loc = pd.concat(self.n, new_df.set_index('node'))
        else:
             self.n.loc = pd.condat(self.n, new_df.set_index('node'))

    def add_scenarios(self, wind_scenario, pv_scenario, hydrogen_demand_scenario, elec_load_scenario):
        n_scenarios = self.n_scenarios
        self.genW_t = wind_scenario.sel(scenario=slice(0,n_scenarios-1))
        self.genS_t = pv_scenario.sel(scenario=slice(0,n_scenarios-1))
        self.loadH_t = hydrogen_demand_scenario.sel(scenario=slice(0,n_scenarios-1)) 
        self.loadP_t = elec_load_scenario.sel(scenario=slice(0,n_scenarios-1)) 

        self.T = self.genW_t.shape[0]
        self.d = self.genW_t.shape[2]
        
    def init_time_partition(self):
        """
        Initializes the time partition for the Network.
        This method creates a `time_partition` object using the `time_partition` class and assigns it to the `self.time_partition` attribute. It then retrieves the aggregation specification from the `agg` attribute of the `time_partition` object.
        The method then uses the `df_aggregator` function to aggregate the `genW_t`, `genS_t`, `loadH_t`, and `loadP_t` attributes of the `self` object based on the aggregation specification. The aggregated data is assigned to the corresponding attributes of the `self` object.
        """
        self.time_partition = time_partition(self.T, self.init_method)
        agg = self.time_partition.agg
        self.genW_t_agg = df_aggregator(self.genW_t, agg)
        self.genS_t_agg = df_aggregator(self.genS_t, agg)
        self.loadH_t_agg = df_aggregator(self.loadH_t, agg)
        self.loadP_t_agg = df_aggregator(self.loadP_t, agg)
   
    #network, df0, df, splitted_intervals, son_indeces_lists
    def update_time_partition(self):
        tp_obj = self.time_partition
        tp = tp_obj.agg
        splitted_intervals = tp_obj.family_tree[-1]
        son_indeces_lists = [tp_obj.interval_subsets(father_interval, tp) for father_interval in splitted_intervals]

        df0 = self.genW_t
        df = self.genW_t_agg
       
        self.genW_t_agg = df_aggregator2(self, self.genW_t, self.genW_t_agg, splitted_intervals, son_indeces_lists)
        self.genS_t_agg =df_aggregator2(self,self.genS_t, self.genS_t_agg, splitted_intervals, son_indeces_lists)
        self.loadH_t_agg = df_aggregator2(self,self.loadH_t, self.loadH_t_agg, splitted_intervals, son_indeces_lists)
        self.loadP_t_agg = df_aggregator2(self,self.loadP_t, self.loadP_t_agg, splitted_intervals, son_indeces_lists)

    def iter_partition(self,k=1):
        self.time_partition.iter_partition(k)
        self.update_time_partition()

    def rho_iter_partition(self,VARS):
        "a posteriori iteration method"
        rhoP, rhoH, varho = get_rho(self, VARS)
        varho_grpd =varho.groupby('interval').sum()#drop singletons intervals
        varho_grpd = varho_grpd.where(varho_grpd['interval'].isin([k for k in range(len(tp)) if type(tp[k]) is list]), drop = True) 
        top_n_intervals = xr_top_n(varho_grpd, 10, dim='interval')
        self.time_partition.iter_partition_intervals(top_n_intervals)
        self.update_time_partition()

    def plot(self):
        """
        Plot the network using folium.
        """
        if 'node' in self.n.columns:
            self.n.set_index('node', inplace=True)
        # Filter out rows with NaN values in 'lat' or 'long'
        valid_nodes = self.n.dropna(subset=['lat', 'long'])
        print("valid nodes",valid_nodes.index)
        loc = [valid_nodes['lat'].mean(), valid_nodes["long"].mean()]
        m = folium.Map(location=loc, zoom_start=5, tiles='CartoDB positron')
        
        for node in valid_nodes.index.to_list():
            folium.Marker(
                location=(valid_nodes.loc[node, 'lat'], valid_nodes.loc[node, 'long']),
                icon=folium.Icon(color="green"),
            ).add_to(m)
        
        for edge in self.edgesP.index.to_list():
            start_node, end_node = self.edgesP.loc[edge, 'start_node'], self.edgesP.loc[edge, 'end_node']
            if start_node in valid_nodes.index and end_node in valid_nodes.index:
                start_loc = (valid_nodes.loc[start_node, 'lat'], valid_nodes.loc[start_node, 'long'])
                end_loc = (valid_nodes.loc[end_node, 'lat'], valid_nodes.loc[end_node, 'long'])
                folium.PolyLine([start_loc, end_loc], weight=5, color='blue', opacity=.2).add_to(m)
            else:
                print("node not valid edge",start_node,end_node)
                print(self.n)
        
        self.n.reset_index(inplace=True)
        return m._repr_html_()

    def get_edgesP(self,start_node,end_node):
        return self.edgesP[self.edgesP['start_node']==start_node][self.edgesP['end_node']==end_node]

    def get_edgesH(self,start_node,end_node):
        return self.edgesH[self.edgesH['start_node']==start_node][self.edgesH['end_node']==end_node]
    

    def neighbour_edgesP(self,direction, node_index):
        """
        Returns a list of edge indices that are connected to a given node in a specified direction.

        Parameters:
            direction (str): The direction of the edges to be retrieved. '+' for outgoing edges and '-' for incoming edges.
            node_index (int): The index of the node in the graph.

        Returns:
            list: A list of edge indices that are connected to the given node in the specified direction.
        """
        if direction == '+':
            return self.edgesP.loc[self.edgesP['start_node']==self.n.index.to_list()[node_index]].index.to_list()
        if direction == '-':
            return self.edgesP.loc[self.edgesP['end_node']==self.n.index.to_list()[node_index]].index.to_list()

    def neighbour_edgesH(self,direction, node_index):
        if direction == '+':
            return self.edgesH.loc[self.edgesH['start_node']==self.n.index.to_list()[node_index]].index.to_list()
        if direction == '-':
            return self.edgesH.loc[self.edgesH['end_node']==self.n.index.to_list()[node_index]].index.to_list()

    def to_dict(self):
        """
        Save the Network object as a dictionary of dictionaries, where each attribute is stored as a dictionary.

        Returns:
            dict: A dictionary containing all the attributes of the Network object.
        """
        output = {}
        if 'node' in self.n.columns:
            self.n.set_index('node', inplace=True)
        for attr in self.__dict__:
            attribute = getattr(self, attr)
            if type(attribute) is int:
                output[attr] = attribute
            elif type(attribute) is float:
                output[attr] = attribute
            elif type(attribute) is str:
                output[attr] = attribute
            elif type(attribute) is dict:
                output[attr] = attribute
            elif type(attribute) is list:
                output[attr] = attribute
            elif attr == 'time_partition':
                output[attr] = attribute.to_dict()
            else:
                output[attr] = getattr(self, attr).to_dict()
        return output
    
    @staticmethod
    def from_dict(dictionary):
        '''
        converts a dictionary to a Network object
        '''
        n = Network()
        for attr, value in dictionary.items():
            if type(value) is int:
                setattr(n, attr, value)
            elif type(value) is float:
                setattr(n, attr, value)
            elif type(value) is str:
                setattr(n, attr, value)
            elif type(value) is list:
                setattr(n, attr, value)
            elif attr == 'time_partition':
                tp = time_partition.from_dict(value)
                setattr(n, attr, tp)
            elif type(value) is dict:
                if len(value) == 0:
                    setattr(n, attr, pd.DataFrame(value))
                elif attr in ["n","edgesP","edgesH","costs"]:
                    #print(f"it should become a dataframe but let's see, {attr}")
                    max_len = max([len(val) for key, val in value.items()])
                    for key, val in value.items():
                        if len(val) < max_len:
                            value[key] = np.concatenate((val, np.full((max_len - len(val)),np.nan)))
                    setattr(n, attr, pd.DataFrame(value))     
                else:
                    #print(f"it should become a dataarray but let's see, {attr}")
                    da = xr.DataArray.from_dict(value)
                    setattr(n, attr, da)
            else:
                print(f"Some attribute is not supported: {attr} has value of type: {type(value)}")
        
        #n.n = n.n.set_index('node')
        if type(n.edgesP.index[0]) == str:
            n.edgesP.index = n.edgesP.index.astype(int)
            n.edgesH.index = n.edgesH.index.astype(int)

        return n



# %%
