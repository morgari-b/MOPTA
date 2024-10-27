# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:20:20 2024

@author: frulcino
"""

# %%
#%%
import os
# Change to the desired directory
print("Ciao Bia :) se il codice non ti va a random prova a cambiare directory:")
os.chdir('C:/Users/ghjub/codes/MOPTA')
#%%
import numpy as np
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates
import os
import copy
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray, import_scenario_val
from model.EU_net import EU
#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
#import plotly.express as px
import plotly.graph_objects as go
from model.OPT_methods import OPT_agg_correct, OPT3, OPT_agg2
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#%% LOAD FILES
def load_results(file_path, method, name, ext = '.npy', is_list_of_lists = False):
    name = file_path+method+name+ext
    res = np.load(name, allow_pickle=True)
    if is_list_of_lists:
        costs = [[res[i][j]['obj'] for j in range(1,len(res[i]))] for i in range(len(res))]
        times = [[res[i][j]['opt_time'] for j in range(1,len(res[i]))] for i in range(len(res))]
    else:
        costs = [res[i]['obj'] for i in range(len(res))]
        times = [res[i]['opt_time'] for i in range(1,len(res))]
    return res, costs, times

def recover_attributes(res,attr, is_list_of_lists = False):
    if is_list_of_lists:
        L = [[res[i][j][attr] for j in range(1,len(res[i]))] for i in range(len(res))]
    else:
        L = [res[i][attr] for i in range(len(res))]
    return L

    
#%%
file_path = "../saved_opt/"
name = "40iter3"
ext = '.npy'

res_val, costs_val, times_val = load_results(file_path, 'vars_val2', name, ext)
res_rho, costs_rho, times_rho = load_results(file_path, 'vars_rho', name, ext)
#%%
res_random, costs_random, times_random = load_results(file_path, 'vars_random', name, ext, is_list_of_lists = True)

# %%
N_random = len(costs_random)
fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_rho)), y = times_rho, name = 'rho'),
                     #go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     go.Scatter(x = np.arange(len(costs_val)), y = times_val, name = 'validation'),
                     #go.Scatter(x = np.arange(len(costs_val3)), y = costs_val3, name = 'validation3'),
                     #go.Scatter(x = np.arange(len(costs_val_old)), y = costs_val_old, name = 'validation old'),
                     go.Scatter(x = np.arange(len(times_random[0])), y = [np.mean([times_random[i][j] for i in range(N_random)]) for j in range(len(costs_random[0]))], name = 'random')
                     ])
fig.update_layout(title='Total iteration time over iteration', xaxis_title='iteration', yaxis_title='time (s) ')
fig.show()
# %%
times_val_diff = [times_val[i]-times_val[i-1] for i in range(1,len(times_val))]
times_random_diff = [[times_random[i][j]-times_random[i][j-1] for j in range(1,len(costs_random[0]))] for i in range(N_random)]
times_rho_diff = [times_rho[i]-times_rho[i-1] for i in range(1,len(times_rho))]
# %%
fig = go.Figure(data=[go.Scatter(x = np.arange(1,len(times_val)), y = times_val_diff, name = 'validation'),
                     go.Scatter(x = np.arange(len(times_random[0])), y = [np.mean([times_random_diff[i][j] for i in range(N_random)]) for j in range(len(costs_random[0])-1)], name = 'random'),
                     go.Scatter(x = np.arange(1,len(times_rho)), y = times_rho_diff, name = 'rho')
                     ])
fig.update_layout(title=' Iteration time over iteration', xaxis_title='iteration number', yaxis_title='time (s)')
fig.show()

# %%
#plots
print("comment out lines that are not needed")
fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     #go.Scatter(x = np.arange(len(costs_rho)), y = costs_rho, name = 'rho'),
                     #go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     #go.Scatter(x = np.arange(len(costs_val3)), y = costs_val3, name = 'validation3'),
                     #go.Scatter(x = np.arange(len(costs_val_old)), y = costs_val_old, name = 'validation old'),
                     go.Scatter(x = np.arange(len(costs_random[0])), y = [np.mean([costs_random[i][j] for i in range(N_random)]) for j in range(len(costs_random[0]))], name = 'random')
                     ])
fig.update_layout(title='Objective value over iterations', xaxis_title='iteration number', yaxis_title='cost (€) ')
fig.show()
# %%
nh_val =  np.stack(recover_attributes(res_val,'nh', is_list_of_lists = False))
nh_rho =  np.stack(recover_attributes(res_rho,'nh', is_list_of_lists = False))
nh_random0 = recover_attributes(res_random,'nh', is_list_of_lists = True)
nh_random = [np.stack(nh_random0[i]) for i in range(N_random)]


#%%

fig = go.Figure(data=[go.Scatter(x = np.arange(len(nh_val)), y = nh_val[:,0], name = 'validation'),
                     go.Scatter(x = np.arange(len(nh_rho)), y = nh_rho[:,0], name = 'rho'),
                     go.Scatter(x = np.arange(len(nh_random[0])), y = [np.mean([nh_random[i][j,0] for i in range(N_random)]) for j in range(len(nh_random[0]))], name = 'average_random')
                     ])
fig.update_layout(title='nh over iterations', xaxis_title='iteration', yaxis_title='nh')
fig.show()
# %%
