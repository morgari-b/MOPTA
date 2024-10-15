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
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray, import_scenario_val
from model.EU_net import EU
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from model.OPT_methods import OPT_agg_correct, OPT3

#%%


eu=EU()
VARS=OPT_agg_correct(eu)
VARS = VARS[0]
# %%

n = eu

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
#%%
#for t in I:
Pnett = PL - nw*EW - ns*ES 
Hnett = HL #we don't haev any hydrogen generations corresponding to time independent variables
    #P_net.append(Pnett)
    #H_net.append(Hnett)
    


# %%
