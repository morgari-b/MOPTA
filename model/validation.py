#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:20:20 2024

@author: morgari
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


# %% trials

eu=EU()
VARS=OPT_agg_correct(eu)

#%%
scen = import_scenario_val(1,10)

#%%
scenarios = import_scenario_val(6,20)

#%%
print("fetching scenarios from network")
scenarios = {}
scenarios['wind_scenario'] = eu.genW_t
scenarios['pv_scenario'] = eu.genS_t
scenarios['hydrogen_demand_scenario'] = eu.loadH_t
scenarios['elec_load_scenario'] = eu.loadP_t
#%%
Hs2 =Validate(eu,VARS,scenarios)

#%%
eu1 = EU(n_scenarios=1)
VARS1 = OPT3(eu1)

#%%
Hs = Validate(eu1,VARS1,scenarios)


# %% debugging
