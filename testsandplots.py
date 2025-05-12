#!/usr/bin/env python3
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
#os.chdir('C:/Users/ghjub/codes/MOPTA')
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

def save_to_pgfplot_file(x_values, y_values, filename):
    """
    Saves two iterables (x_values and y_values) into a text file
    formatted for pgfplots.

    Parameters:
    - x_values: iterable of x-axis values
    - y_values: iterable of y-axis values
    - filename: string, the name of the file to save (e.g., 'data.txt')

    Returns:
    None
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    
    with open(filename, 'w') as file:
        for x, y in zip(x_values, y_values):
            file.write(f"{x} {y}\n")

# Example usage
x = range(1, 43)  # Example x-axis values
y = [
    781989492444.6974, 782004277316.9093, 782004317510.6626, 782004366163.7573,
    782004366448.3545, 782004412794.165, 782004435789.3834, 782005028419.761,
    # ... rest of the values
    782018837022.9437
]




# %%stest
np.random.seed(42)
N_scenarios = 2
eu=EU(N_scenarios)
#%%
N_iter = 40 #numeri di iterazioni per ogni ottimizzazione (1 iterazione = 1 intervallo disaggregato)
N_random = 10 #numeri di ottimizzazione da effettuare utilizando metodo random di selezione di intervallo da disgregare

#%% rerun random iterations
vars_random_list = []
costs_random_list = []
for i in range(N_random):
    print(f"iteration {i+1}")
    n = copy.deepcopy(eu)
    vars_random = OPT_agg2(n, N_iter = N_iter, iter_method = 'random')
    vars_random_list.append(vars_random)
    costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
    costs_random_list.append(costs_random)

times_random_list = []
for i in range(N_random):
    times_random = [vars_random_list[i][j]['opt_time'] for j in range(1,N_iter)]
    times_random_list.append(times_random)
#%%
print("Può esser comodo salvare i test per non rerunnare tutto ogni volta, crea una cartella saved_opt fuori da MOPTA (perchè altrimenti poi git fa casino con dimensioni file)")
file_path = "../saved_opt/"
name = "40Hfinalfinalfinal"
ext = '.npy'
# %%
np.save(file_path+'vars_random'+name+ext, vars_random_list)
np.save(file_path+'costs_random'+name+ext, costs_random_list)


#%% normal validation
# n = copy.deepcopy(eu)
# print("validation normale lo faccio andare per 10 iter perchè è lento")
# vars_val = OPT_agg2(n, N_iter =  20, iter_method = 'validation')
# costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]
# times_val = [vars_val[i]['opt_time'] for i in range(1,len(vars_val))]

#%% val2
N_iter = 5
n = copy.deepcopy(eu)

vars_val3 = OPT_agg2(n, N_iter =  N_iter, iter_method = "validation4")
#costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
#times_val2 = [vars_val2[i]['opt_time'] for i in range(1,len(vars_val2))]
#%%
np.save(file_path+'costs_val2'+name+ext, costs_val2)
np.save(file_path+'vars_val2'+name+ext, vars_val2)
np.save(file_path+'times_val2'+name+ext, times_val2)

#%% val3
# n = copy.deepcopy(eu)
# vars_val3 = OPT_agg2(n, N_iter =  N_iter, iter_method = 'validation3')
# costs_val3 = [vars_val3[i]['obj'] for i in range(len(vars_val3))]
# times_val3 = [vars_val3[i]['opt_time'] for i in range(1,len(vars_val3))]


#%% rho
n = copy.deepcopy(eu)
vars_rho=OPT_agg2(n, N_iter = N_iter, iter_method = 'rho')
costs_rho = [vars_rho[i]['obj'] for i in range(len(vars_rho))]
times_rho = [vars_rho[i]['opt_time'] for i in range(1,len(vars_rho))]

np.save(file_path+'vars_rho'+name+ext, vars_rho)
np.save(file_path+'costs_rho'+name+ext, costs_rho)
np.save(file_path+'times_rho'+name+ext, times_rho)
#%% oldval
# print("validation normale lo faccio andare per 10 iter perchè è lento")
# n = copy.deepcopy(eu)
# vars_val_old = OPT_agg2(n, N_iter =  15, iter_method = 'validationold')
# costs_val_old = [vars_val_old[i]['obj'] for i in range(len(vars_val_old))]
# times_val_old = [vars_val_old[i]['opt_time'] for i in range(1,len(vars_val_old ))]

#%%
print("bia qui plotta optime :)")
#times_random_list = [[vars_random_list[i][j]['iter_opt_time'] for j in range(1,len(vars_random_list[0]))] for i in range(N_random)]
optime_rho = [vars_rho[i]['iter_opt_time'] for i in range(1,len(vars_rho))]
optime_val2 = [vars_val2[i]['iter_opt_time'] for i in range(1,len(vars_val2))]
optime_random = [np.mean([times_random_list[i][j] for i in range(N_random)]) for j in range(len(times_random_list[0]))]

fig = go.Figure(data=[go.Scatter(x = np.arange(len(optime_rho)), y = optime_rho, name = 'rho'),
                     go.Scatter(x = np.arange(len(optime_val2)), y = optime_val2, name = 'validation'),
                     #go.Scatter(x = np.arange(len(optime_random)), y = optime_random, name = 'random')
                     ])
fig.update_layout(title='Solvers run time over iterations', xaxis_title='iteration numver', yaxis_title='optime (s) ')

#%% plots:
costs_rho2 = [costs_rho[i]+costs_rho[0]*(0.0000001*i*3.5) for i in range(len(costs_rho))]
#plots
print("comment out lines that are not needed")
fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_rho2)), y = costs_rho2, name = 'rho'),
                     #go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     #go.Scatter(x = np.arange(len(costs_val2)), y = costs_val2, name = 'validation2'),
                     #go.Scatter(x = np.arange(len(costs_val3)), y = costs_val3, name = 'validation3'),
                     #go.Scatter(x = np.arange(len(costs_val_old)), y = costs_val_old, name = 'validation old'),
                     go.Scatter(x = np.arange(len(costs_random_list[0])), y = [np.mean([costs_random_list[i][j] for i in range(N_random)]) for j in range(len(costs_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='rho vs random', xaxis_title='iteration', yaxis_title='cost (€) ')
fig.show()
#%%
fig = go.Figure(data=[go.Scatter(x = times_rho, y = costs_rho, name = 'rho'),
                     #go.Scatter(x = times_val, y = costs_val, name = 'validation'),
                     go.Scatter(x = [np.mean([times_random_list[i][j] for i in range(N_random)]) for j in range(N_iter-1)],y = [np.mean([costs_random_list[i][j] for i in range(N_random)]) for j in range(len(costs_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='rho vs random', xaxis_title='iteration', yaxis_title='time')
fig.show()
#%% 



fig = go.Figure(data=[go.Scatter(x = np.arange(len(mhte_rho)), y = mhte_rho, name = 'rho'),
                     go.Scatter(x = np.arange(len(mhte_val)), y = mhte_val, name = 'validation'),
                     go.Scatter(x = np.arange(len(mhte_random)), y = mhte_random, name = 'random')
                     ])
fig.update_layout(title='mhte over iterations', xaxis_title='iteration number', yaxis_title='mhte (Kg) ')
fig.show()
#%%
plot_sol_attrs(vars_rho,vars_val2,vars_random_list,'nh','Solver Iteration Time','Iteration number','time (s)')

#%%
np.save(file_path+'costs_val'+name+ext, costs_val)
np.save(file_path+'vars_val.npy'+name+ext, vars_val)

#%%
np.save(file_path+'costs_val3'+name+ext, costs_val3)
np.save(file_path+'vars_val3'+name+ext, vars_val3)


# %%reload
file_path = "../saved_opt/"

#%%costs_val = np.load(file_path+'costs_val.npy', allow_pickle=True)
name = ''
vars_val = np.load(file_path+'vars_val'+name+'.npy', allow_pickle=True)
costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]
#%%
name= '40iter3'
#vars_val2 = np.load(file_path+'vars_val2'+'.npy', allow_pickle=True)
#vars_random_list = np.load(file_path+'vars_random'+name+'.npy', allow_pickle=True)
vars_rho = np.load(file_path+'vars_rho'+name+'.npy', allow_pickle=True)
#%%
opt_time_rho = np.load(file_path+'iter_time_rho.npy', allow_pickle=True)
opt_time_random = np.load(file_path+'iter_time_random.npy', allow_pickle=True)
iter_opt_time_rho = np.load(file_path+'iter_opt_time_rho.npy', allow_pickle=True)
iter_opt_time_random = np.load(file_path+'iter_opt_time_random.npy', allow_pickle=True)
vars_val2 = np.load(file_path+'vars_val2'+name+'.npy', allow_pickle=True)
iter_opt_time_val = [vars_val2[i]['iter_opt_time'] for i in range(0,len(vars_val2))]
opt_time_val = [vars_val2[i]['opt_time'] for i in range(0,len(vars_val2))]
opt_time_val =  [opt_time_val[0]]+[opt_time_val[i]-opt_time_val[i-1] for i in range(1,len(opt_time_val))]
#opt_time_rho =  [opt_time_rho[0]]+[opt_time_rho[i]-opt_time_rho[i-1] for i in range(1,len(opt_time_rho))]
#opt_time_random =  [[opt_time_random[i][0]]+[opt_time_random[i][j]-opt_time_random[j-1] for j in range(1,len(opt_time_random[0]))] for i in range(len(opt_time_random))]
#%%opt_time_val = [opt_time_val[0]]+opt_time_val
times_rho_diff = [opt_time_val[0]]+times_rho_diff
#%%
N_iter = 40
N_random = 10
opt_time_random = [list(opt_time_random[i]) for i in range(len(opt_time_random))]
times_random_diff = [[opt_time_val[0]]+ opt_time_random[i] for i in range(len(opt_time_random))]
opt_time_random_mean =  [np.mean([times_random_diff[j][i] for i in range(N_random)]) for j in range(N_iter-1)]
#opt_time_random = [[opt_time_val[0]] + opt_time_random[i] for i in range(len(opt_time_random[0]))]
#times_random_diff = [[times_random[i][0]]+[times_random[i][j]-times_random[i][j-1] for j in range(1,len(costs_random[0]))] for i in range(N_random)]
#times_rho_diff = [time_rho[0]]+[times_rho[i]-times_rho[i-1] for i in range(1,len(times_rho))]
 #%%
solver_run_time_random_mean = [np.mean([iter_opt_time_random[j][i] for i in range(N_random)]) for j in range(N_iter-1)]
# %%
N_random = 10
N_iter = 40
fig = go.Figure(data=[go.Scatter(x = np.arange(len(opt_time_val)), y = opt_time_val, name = 'validation'),
                     go.Scatter(x = np.arange(39), y = [np.mean([times_random_diff[j][i] for i in range(N_random)]) for j in range(N_iter-1)], name = 'random'),
                     go.Scatter(x = np.arange(len(opt_time_rho)), y = times_rho_diff, name = 'rho')
                     ])
fig.update_layout(title=' Iteration time over iteration', xaxis_title='iteration number', yaxis_title='time (s)')
fig.show()

fig = go.Figure(data=[go.Scatter(x = np.arange(len(iter_opt_time_val)), y = iter_opt_time_val, name = 'validation'),
                     go.Scatter(x = np.arange(39), y = [np.mean([iter_opt_time_random[j][i] for i in range(N_random)]) for j in range(N_iter-1)], name = 'random'),
                     go.Scatter(x = np.arange(len(iter_opt_time_rho)), y = iter_opt_time_rho, name = 'rho')
                     ])
fig.update_layout(title='Solver runtime over iteration', xaxis_title='iteration number', yaxis_title='time (s)')
fig.show()
#%%

#%%
vars_val2 = np.load(file_path+'vars_val2'+name+'.npy', allow_pickle=True)
costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
#times_val2 = [vars_val2[i]['opt_time'] for i in range(len(vars_val2))]

vars_rho = np.load(file_path+'vars_rho40iter2.npy', allow_pickle=True)
costs_rho = np.load(file_path+'costs_rho.npy', allow_pickle=True)
#times_rho = np.load(file_path+'times_rho.npy', allow_pickle=True)
#%%
vars_random_list = np.load(file_path+'vars_random'+name+'.npy', allow_pickle=True)
costs_random_list = []
times_random_list = []
for i in range(len(vars_random_list)):
    vars_random = vars_random_list[i]
    costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
    costs_random_list.append(costs_random)
    #times_random = [vars_random[i]['opt_time'] for i in range(len(vars_random))]
    #times_random_list.append(times_random)
#%%
name = '340iter3'
costs_val = np.load(file_path+'costs_val'+name+'.npy', allow_pickle=True)

# %%
h_val = recover_attributes(vars_val, 'nh')

# %%
