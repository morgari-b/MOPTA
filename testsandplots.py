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

# %%stest
np.random.seed(42)
N_scenarios = 2
eu=EU(N_scenarios)

N_iter = 20 #numeri di iterazioni per ogni ottimizzazione (1 iterazione = 1 intervallo disaggregato)
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


#%% normal validation
n = copy.deepcopy(eu)
print("validation normale lo faccio andare per 10 iter perchè è lento")
vars_val = OPT_agg2(n, N_iter =  10, iter_method = 'validation')
costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]
times_val = [vars_val[i]['opt_time'] for i in range(1,len(vars_val))]

#%% val2
n = copy.deepcopy(eu)
vars_val2 = OPT_agg2(n, N_iter =  N_iter, iter_method = 'validation2')
costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
times_val2 = [vars_val2[i]['opt_time'] for i in range(1,len(vars_val2))]
#%% val3
n = copy.deepcopy(eu)
vars_val3 = OPT_agg2(n, N_iter =  N_iter, iter_method = 'validation3')
costs_val3 = [vars_val3[i]['obj'] for i in range(len(vars_val3))]
times_val3 = [vars_val3[i]['opt_time'] for i in range(1,len(vars_val3))]


#%% rho
n = copy.deepcopy(eu)
vars_rho=OPT_agg2(n, N_iter = N_iter, iter_method = 'rho')
costs_rho = [vars_rho[i]['obj'] for i in range(len(vars_rho))]
times_rho = [vars_rho[i]['opt_time'] for i in range(1,len(vars_rho))]

#%% oldval
print("validation normale lo faccio andare per 10 iter perchè è lento")
n = copy.deepcopy(eu)
vars_val_old = OPT_agg2(n, N_iter =  8, iter_method = 'validationold')
costs_val_old = [vars_val_old[i]['obj'] for i in range(len(vars_val_old))]
times_val_old = [vars_val_old[i]['opt_time'] for i in range(1,len(vars_val_old ))]


#%% plots:

#plots
print("comment out lines that are not needed")
fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_rho)), y = costs_rho, name = 'rho'),
                     go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     go.Scatter(x = np.arange(len(costs_val2)), y = costs_val2, name = 'validation2'),
                     go.Scatter(x = np.arange(len(costs_val3)), y = costs_val3, name = 'validation3'),
                     go.Scatter(x = np.arange(len(costs_val_old)), y = costs_val_old, name = 'validation old'),
                     go.Scatter(x = np.arange(len(costs_random_list[0])), y = [np.mean([costs_random_list[i][j] for i in range(N_random)]) for j in range(len(costs_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='rho vs random', xaxis_title='iteration', yaxis_title='cost (€) ')
fig.show()
#%%
fig = go.Figure(data=[go.Scatter(x = times_rho, y = costs_rho, name = 'rho'),
                     go.Scatter(x = times_val, y = costs_val, name = 'validation'),
                     go.Scatter(x = [np.mean([times_random_list[i][j] for i in range(N_random)]) for j in range(N_iter-1)],y = [np.mean([costs_random_list[i][j] for i in range(N_random)]) for j in range(len(costs_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='rho vs random', xaxis_title='iteration', yaxis_title='time')
fig.show()
#%% 




#%%
print("Può esser comodo salvare i test per non rerunnare tutto ogni volta, crea una cartella saved_opt fuori da MOPTA (perchè altrimenti poi git fa casino con dimensioni file)")
file_path = "../saved_opt/"
name = "40iter"
ext = '.npy'

#%%
np.save(file_path+'costs_val'+name+ext, costs_val)
np.save(file_path+'vars_val.npy'+name+ext, vars_val)
#%%
np.save(file_path+'costs_val2'+name+ext, costs_val)
np.save(file_path+'vars_val2'+name+ext, vars_val)
# %%
np.save(file_path+'vars_random'+name+ext, vars_random_list)
np.save(file_path+'costs_random'+name+ext, costs_random_list)

#%%
np.save(file_path+'vars_rho'+name+ext, vars_rho)
np.save(file_path+'costs_rho'+name+ext, costs_rho)
# %%reload
file_path = "../saved_opt/"

#%%costs_val = np.load(file_path+'costs_val.npy', allow_pickle=True)
vars_val = np.load(file_path+'vars_val.npy', allow_pickle=True)
costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]
#%%

#%%
vars_val2 = np.load(file_path+'vars_val2.npy', allow_pickle=True)
costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
times_val2 = [vars_val2[i]['opt_time'] for i in range(len(vars_val2))]

vars_rho = np.load(file_path+'vars_rho.npy', allow_pickle=True)
costs_rho = np.load(file_path+'costs_rho.npy', allow_pickle=True)
times_rho = np.load(file_path+'times_rho.npy', allow_pickle=True)
#%%
vars_random_list = np.load(file_path+'vars_random.npy', allow_pickle=True)
costs_random_list = []
times_random_list = []
for i in range(len(vars_random_list)):
    vars_random = vars_random_list[i]
    costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
    costs_random_list.append(costs_random)
    #times_random = [vars_random[i]['opt_time'] for i in range(len(vars_random))]
    #times_random_list.append(times_random)



# %%


# %%
