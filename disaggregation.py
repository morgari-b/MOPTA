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
from model.OPT_methods import OPT_agg_correct, OPT3, OPT_agg2

#%%


eu=EU()
n = copy.deepcopy(eu)
#%%
vars_rho=OPT_agg2(n, N_iter = 10, iter_method = 'rho')
costs_rho = [vars_rho[i]['obj'] for i in range(len(vars_rho))]
# %%
vars_random_list = []
costs_random_list = []
#%%
for i in range(10):
    n = copy.deepcopy(eu)
    vars_random = OPT_agg2(n, N_iter = 10, iter_method = 'random')
    vars_random_list.append(vars_random)
    costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
    costs_random_list.append(costs_random)

# %%
