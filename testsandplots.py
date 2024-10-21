#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:20:20 2024

@author: frulcino
"""

# %%
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

#%%

N_scenarios = 2
eu=EU(N_scenarios)

N_iter = 10 #numeri di iterazioni per ogni ottimizzazione (1 iterazione = 1 intervallo disaggregato)
N_random = 10 #numeri di ottimizzazione da effettuare utilizando metodo random di selezione di intervallo da disgregare
#%%
# n = copy.deepcopy(eu)
# vars_rho=OPT_agg2(n, N_iter = N_iter, iter_method = 'rho')
# costs_rho = [vars_rho[i]['obj'] for i in range(len(vars_rho))]
#%%
n = copy.deepcopy(eu)
vars_val = OPT_agg2(n, N_iter =  N_iter, iter_method = 'validation')
costs_val = [vars_val[i]['obj'] for i in range(len(vars_val))]

vars_random_list = []
costs_random_list = []

for i in range(N_random):
    n = copy.deepcopy(eu)
    vars_random = OPT_agg2(n, N_iter = 10, iter_method = 'random')
    vars_random_list.append(vars_random)
    costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
    costs_random_list.append(costs_random)


fig = go.Figure(data=[go.Scatter(x = np.arange(len(costs_val)), y = costs_rho, name = 'rho'),
                     go.Scatter(x = np.arange(len(costs_random_list[0])), y = [np.mean([costs_random_list[i][j] for i in range(5)]) for j in range(len(costs_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='validation vs random', xaxis_title='iteration', yaxis_title='cost')
fig.show()

#times_rho = [vars_rho[i]['opt_time'] for i in range(len(vars_rho))]
time_val = [vars_val[i]['opt_time'] for i in range(len(vars_val))]
times_random_list = []
for i in range(N_random):
    times_random = [vars_random_list[i][j]['opt_time'] for j in range(len(vars_random_list[i]))]
    times_random_list.append(times_random)


fig = go.Figure(data=[go.Scatter(x = np.arange(len(times_rho)), y = times_rho, name = 'rho'),
                     go.Scatter(x = np.arange(len(times_random_list[0])), y = [np.mean([times_random_list[i][j] for i in range(5)]) for j in range(len(times_random_list[0]))], name = 'average_random')
                     ])
fig.update_layout(title='rho vs random', xaxis_title='iteration', yaxis_title='time')
fig.show()






# %%
