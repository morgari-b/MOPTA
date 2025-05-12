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
from model.validation import Validate, Validate2, ValidateHfix, import_scenario_val
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario, scenario_to_array
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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


# %%
