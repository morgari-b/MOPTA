#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:05:49 2024v
@author: morgari
CURRENT WORKING MODELS:
"""
# %%

# import sys
# import os

# # Add the parent directory of 'model' to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#%%
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandapower.networks as pn
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates
import os
#%% os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")
import xarray as xr
import folium
from model.time_partition import time_partition
from model.aggregator import df_aggregator2, df_aggregator
from model.validation import Validate, Validate2, ValidateHfix, import_scenario_val
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario, scenario_to_array
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



def get_rho(n,VARS, n_max = 10):
    """
    given a network and a solution to an aggregated problem, calculates the rho value for each constraint
    """
    #rho for a constraint
    tp = n.time_partition.agg
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
    pt = n.time_partition.partitionize(tp)
    intervals = [i for L in [[k]*len(pt[k]) for k in range(len(tp))] for i in L]
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

        def __str__(self):
            # Customize the string representation of the Network object
            return (
                f"Network with {len(self.n)} nodes, "
                f"{len(self.edgesP)} power edges, "
                f"{len(self.edgesH)} hydrogen edges, "
                f"{self.n_scenarios} scenarios,"
                f"{self.T} time steps and "
                f"{len(self.time_partition.agg)} aggregated time steps."
                #f"{self.loadP_t.mean()} {self.loadH_t.mean()} {self.genW_t.mean()} {self.genS_t.mean()} mean of power and hydrogen demand, wind and solar generation"
            )

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

        self.T = self.genW_t.time.shape[0]
        self.d = self.genW_t.scenario.shape[0]

    def add_date_to_agg_array(self,H, init_date = pd.Timestamp(2023, 1, 1)):
        tp = self.time_partition.agg
        #print(tp)
        tp0 = [tp[i][0] if type(tp[i]) is list else tp[i] for i in range(len(tp))]
        H0 = H.copy().isel(time=tp0)
        dates =[init_date + pd.Timedelta(hours=int(t)) for t in H0.time.values]
        H0.coords['date'] = ('time', dates)
        return H0 

    def init_time_partition(self):
        """
        Initializes the time partition for the Network.
        This method creates a `time_partition` object using the `time_partition` class and assigns it to the `self.time_partition` attribute. It then retrieves the aggregation specification from the `agg` attribute of the `time_partition` object.
        The method then uses the `df_aggregator` function to aggregate the `genW_t`, `genS_t`, `loadH_t`, and `loadP_t` attributes of the `self` object based on the aggregation specification. The aggregated data is assigned to the corresponding attributes of the `self` object.
        """
        self.time_partition = time_partition(self.T, self.init_method)
        agg = self.time_partition.agg
        self.Tagg = len(agg)
        self.genW_t_agg = df_aggregator(self.genW_t, agg)
        self.genS_t_agg = df_aggregator(self.genS_t, agg)
        self.loadH_t_agg = df_aggregator(self.loadH_t, agg)
        self.loadP_t_agg = df_aggregator(self.loadP_t, agg)
   
    #network, df0, df, splitted_intervals, son_indeces_lists
    def update_time_partition(self):
        #print("updating time partition")
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
        #print(f"new dataframe shape: {self.genW_t_agg.shape}, len tp {len(tp)}")

    def iter_partition(self,k=1):
        """
        Iteratively partitions the time intervals in the network.

        Parameters:
            k (int): Number of iterations.

        Returns:
            None
        """
        #print("iterating")
        self.time_partition.random_iter_partition(k)
        self.update_time_partition()

    def rho_iter_partition(self,VARS, k=1):
        #"a posteriori iteration method"
        #print("iterating")
        rhoP, rhoH, varho = get_rho(self, VARS)
        tp = self.time_partition.agg
        varho_grpd =varho.groupby('interval').sum()#drop singletons intervals
        varho_grpd = varho_grpd.where(varho_grpd['interval'].isin([k for k in range(len(tp)) if type(tp[k]) is list]), drop = True) 
        top_n_intervals = xr_top_n(varho_grpd, k, dim='interval')['interval']
        ##print(top_n_intervals)
        self.time_partition.iter_partition_intervals(top_n_intervals)
       #TODO: uncomment 
        self.update_time_partition()

    def fail_int(self, fail):
        """
        This function determines in which time interval the validation failed based on the input fail value.

        Parameters:
            fail (int): The day in which the validation failed.

        Returns:
            None
        """
        tp = self.time_partition.agg
        fail_int = 0
        fail_hour = fail*24
        #find in which time interval the validation failed
        for t in tp:
            if type(t) is list:
                if fail_hour <= t[0] and t[0] < fail_hour + 24: #look for an interval which is in the fail day
                    break
            else:
                if fail_hour + 24 <= t:
                    break
            fail_int += 1

        return fail_int

    def validationfun_iter_partition(self, VARS, k=1):
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t

        fail = Validate(self,[VARS],scenarios) #initial hour of the validation on which it fails
        if fail < 365:
           
            fail_int = self.fail_int(fail)
            logging.debug(f'fail_interval is {fail_int}, {tp[fail_int]} on day {fail}')
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print(f"disaggregating interval {fail_int}")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)

            self.update_time_partition()
            return False
        else:
            return True
    def validationfun_iter_partition_old(self, VARS, k=1):
        """
        Validates the current time partition and updates it based on the validation result.

        Parameters:
            VARS: List of variables required for validation.
            k (int, optional): Number of iterations for random partitioning. Default is 1.

        Returns:
            bool: True if validation is successful, False otherwise. If validation fails,
                the function attempts to find the time interval in which it failed and
                disaggregate it. If no interval is found, it performs random iteration.
        """
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t
        fail = Validate(self,[VARS],scenarios)
        if fail < 365:
            fail_int = 0
            #find in which time interval the validation failed
            for t in tp:
                if type(t) is list:
                    if fail in t:
                        break
                else:
                    if fail == t:
                        break
                fail_int += 1
            
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)
            self.update_time_partition()
            return False
        else:
            return True
    def validation2fun_iter_partition(self, VARS, k=1, day_initial = 0, scenario_initial = 0):
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t

        fail_day, fail_scen = Validate2(self,[VARS],scenarios, day_initial, scenario_initial) #initial hour of the validation on which it fails
        if fail_day < 364 or fail_scen < self.d - 1:
           
            fail_int = self.fail_int(fail_day)
            logging.debug(f'debug: fail_interval is {fail_int}, {tp[fail_int]} on day {fail_day}')
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)

            self.update_time_partition()
            return False, fail_day, fail_scen
        else:
            return True, fail_day, fail_scen

    def validationHfix_iter_partition(self, VARS, k=1, day_initial = 0, scenario_initial = 0):
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t

        fail_day, fail_scen = ValidateHfix(self,[VARS],scenarios, day_initial, scenario_initial) #initial hour of the validation on which it fails
        if fail_day < 364 or fail_scen < self.d - 1:
           
            fail_int = self.fail_int(fail_day)
            #logging.debug(f'debug: fail_interval is {fail_int}, {tp[fail_int]} on day {fail_day}')
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                logging.debug(f'debug: fail_interval is {fail_int}, {tp[fail_int]} on day {fail_day}')
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)

            self.update_time_partition()
            return False, fail_day, fail_scen
        else:
            return True, fail_day, fail_scen

    def total_iter_partition(self):
        """
        Perform a total iteration partition on the network.

        Parameters:
            VARS: Variables required for the partition process.

        This method checks if the current aggregation length is equal to the total number 
        of time steps. If not, it performs a random iteration partition to match the total 
        number of time steps and updates the time partition accordingly.
        """

        self.time_partition.total_disaggregate()
        self.update_time_partition()

    def plot(self, output_file="network_map.html"):
        """
        Plot the network using folium.
        """
        if 'node' in self.n.columns:
            self.n.set_index('node', inplace=True)
        # Filter out rows with NaN values in 'lat' or 'long'
        valid_nodes = self.n.dropna(subset=['lat', 'long'])
        #print("valid nodes",valid_nodes.index)
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
        m.save(output_file)
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
                    ##print(f"it should become a dataframe but let's see, {attr}")
                    max_len = max([len(val) for key, val in value.items()])
                    for key, val in value.items():
                        if len(val) < max_len:
                            value[key] = np.concatenate((val, np.full((max_len - len(val)),np.nan)))
                    setattr(n, attr, pd.DataFrame(value))     
                else:
                    ##print(f"it should become a dataarray but let's see, {attr}")
                    da = xr.DataArray.from_dict(value)
                    setattr(n, attr, da)
            else:
                print(f"Some attribute is not supported: {attr} has value of type: {type(value)}")
        
        #n.n = n.n.set_index('node')
        if type(n.edgesP.index[0]) == str:
            n.edgesP.index = n.edgesP.index.astype(int)
            n.edgesH.index = n.edgesH.index.astype(int)

        return n
#%%

# %%
