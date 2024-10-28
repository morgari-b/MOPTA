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
N_iter = 50
n = copy.deepcopy(eu)

vars_val2 = OPT_agg2(n, N_iter =  N_iter, iter_method = "validation5")
costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
times_val2 = [vars_val2[i]['opt_time'] for i in range(1,len(vars_val2))]
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

#plots
print("comment out lines that are not needed")
fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_rho)), y = costs_rho, name = 'rho'),
                     #go.Scatter(x = np.arange(len(costs_val)), y = costs_val, name = 'validation'),
                     go.Scatter(x = np.arange(len(costs_val2)), y = costs_val2, name = 'validation2'),
                     #go.Scatter(x = np.arange(len(costs_val3)), y = costs_val3, name = 'validation3'),
                     #go.Scatter(x = np.arange(len(costs_val_old)), y = costs_val_old, name = 'validation old'),
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


mhte_random = [3308566.6032981756,
 3859582.4782291995,
 4223854.373252868,
 5054103.104566936,
 5148233.9328136835,
 5407111.612410657,
 5726009.1312797535,
 6157548.554565738,
 6315494.515719863,
 6443086.980842555,
 6612359.429819329,
 6761110.913055216,
 7017359.002575858,
 7321389.959312086,
 7518660.015990103,
 7787557.934352474,
 7845271.101424083,
 7869914.1756432075,
 7963826.127462235,
 8012724.456470311,
 8115607.0004054755,
 8128178.788313104,
 8288488.9425484855,
 8308685.708781014,
 8503603.4530205,
 8643892.764034163,
 8643235.519039994,
 8652626.385951485,
 8702891.017229522,
 8710545.596026894,
 8719062.833155835,
 8719534.453670558,
 8719534.453670612,
 8740907.862986013,
 8766367.274560085,
 8766367.274560051,
 8770159.232284743,
 8823107.00635219,
 8835806.552134894,
 8835806.552134957]

mhte_rho = [3308566.6032981756,
 8427552.316842312,
 8427552.316842336,
 8427552.31684232,
 8427552.316842321,
 8427552.316842325,
 8427552.316842735,
 8593663.782622892,
 8901122.339778358,
 8901122.339778258,
 8901122.33977836,
 8953291.331828868,
 8953291.331828853,
 8953291.331829084,
 8953291.331828885,
 8953291.331829099,
 8953291.331829082,
 8979179.901302546,
 8979179.901302546,
 8974087.058085673,
 8974087.058085315,
 8974087.05808526,
 9104287.76432671,
 9104287.764326703,
 9104287.7643265,
 9104287.76432703,
 9104287.764326714,
 9104287.764326707,
 9104287.764326379,
 9104287.764326712,
 9104287.764326712,
 9100043.728312433,
 9100043.72831269,
 9100043.728312546,
 9100043.728312466,
 9100043.728312813,
 9100043.728312578,
 9100043.7283128,
 9100043.728312762,
 9100043.728312735]

mhte_val = [vars_val2[i]['mhte'][4]*9100043.728312735/vars_val2[-1]['mhte'][4] for i in range(40)]

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
vars_val = np.load(file_path+'vars_val.npy', allow_pickle=True)
costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]

#%%
opt_time_rho = np.load(file_path+'iter_time_rho.npy', allow_pickle=True)
opt_time_random = np.load(file_path+'iter_time_random.npy', allow_pickle=True)
iter_opt_time_rho = np.load(file_path+'iter_opt_time_rho.npy', allow_pickle=True)
iter_opt_time_random = np.load(file_path+'iter_opt_time_random.npy', allow_pickle=True)
iter_opt_time_val = [vars_val2[i]['iter_opt_time'] for i in range(1,len(vars_val2))]
opt_time_val = [vars_val2[i]['opt_time'] for i in range(1,len(vars_val2))]
opt_time_val =  [opt_time_val[0]]+[opt_time_val[i]-opt_time_val[i-1] for i in range(1,len(opt_time_val))]
#opt_time_rho =  [opt_time_rho[0]]+[opt_time_rho[i]-opt_time_rho[i-1] for i in range(1,len(opt_time_rho))]
#opt_time_random =  [[opt_time_random[i][0]]+[opt_time_random[i][j]-opt_time_random[j-1] for j in range(1,len(opt_time_random[0]))] for i in range(len(opt_time_random))]
#%%opt_time_val = [opt_time_val[0]]+opt_time_val
times_rho_diff = [opt_time_val[0]]+times_rho_diff
#%%
opt_time_random = [list(opt_time_random[i]) for i in range(len(opt_time_random))]
times_random_diff = [[opt_time_val[0]]+ opt_time_random[i] for i in range(len(opt_time_random))]
#opt_time_random = [[opt_time_val[0]] + opt_time_random[i] for i in range(len(opt_time_random[0]))]
#times_random_diff = [[times_random[i][0]]+[times_random[i][j]-times_random[i][j-1] for j in range(1,len(costs_random[0]))] for i in range(N_random)]
#times_rho_diff = [time_rho[0]]+[times_rho[i]-times_rho[i-1] for i in range(1,len(times_rho))]
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
vars_val2 = np.load(file_path+'vars_val2.npy', allow_pickle=True)
costs_val2 = [vars_val2[i]['obj'] for i in range(len(vars_val2))]
times_val2 = [vars_val2[i]['opt_time'] for i in range(len(vars_val2))]

vars_rho = np.load(file_path+'vars_rho.npy', allow_pickle=True)
costs_rho = np.load(file_path+'costs_rho.npy', allow_pickle=True)
times_rho = np.load(file_path+'times_rho.npy', allow_pickle=True)
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



# %%
h_val = recover_attributes(vars_val, 'nh')

# %%
