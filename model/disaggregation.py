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
# %%
tp_obj = n.time_partition
tp = tp_obj.agg

rhoP, rhoH, varho = get_rho(n, VARS)
varho_grpd =varho.groupby('interval').sum()#drop singletons intervals
varho_grpd = varho_grpd.where(varho_grpd['interval'].isin([k for k in range(len(tp)) if type(tp[k]) is list]), drop = True) 
top_n_intervals = xr_top_n(varho_grpd, 10, dim='interval')
# %% iter partition

    # %%
