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
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray, import_scenario_val, get_rho, xr_top_n
from model.EU_net import EU
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from model.OPT_methods import OPT_agg_correct, OPT3, OPT_agg2

#%%


eu=EU()
#VARS=OPT_agg_correct(eu)
#VARS = VARS[0]
# %%

n = eu

#%%
res = OPT_agg2(n, 3, iter_method = "rho")


# %%
