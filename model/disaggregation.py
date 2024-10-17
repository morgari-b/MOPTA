#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:20:20 2024

@author: frulcino
"""

# %%
import copy
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
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray, import_scenario_val, get_rho, xr_top_n, df_aggregator2
from model.EU_net import EU
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from model.OPT_methods import OPT_agg_correct, OPT3, OPT_agg2
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#%% confront with rho with random
eu = EU()
n = copy.deepcopy(eu)
resrho = OPT_agg2(n, 10, iter_method = "rho")

resrandom = []

for i in range(1):
    print("iteration number ", i)
    n = copy.deepcopy(eu)
    res = OPT_agg2(n, 10, iter_method = "random")
    resrandom.append(res)
# %%
import copy
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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
n = EU(2)
res = OPT_agg2(n, 10, iter_method = "random")

# %%
costs = [res[i]['obj'] for i in range(len(res))]
# %%test disaggregation, siamo caduti in basso
tp_obj = n.time_partition
tp = tp_obj.agg
splitted_intervals = tp_obj.family_tree[-1]
son_indeces_lists = [tp_obj.interval_subsets(father_interval, tp) for father_interval in splitted_intervals]

df0 = n.genW_t
df = n.genW_t_agg
# %%
genW_t_agg = df_aggregator2(n, n.genW_t, n.genW_t_agg, splitted_intervals, son_indeces_lists)
genS_t_agg =df_aggregator2(n,n.genS_t, n.genS_t_agg, splitted_intervals, son_indeces_lists)
loadH_t_agg = df_aggregator2(n,n.loadH_t, n.loadH_t_agg, splitted_intervals, son_indeces_lists)
loadP_t_agg = df_aggregator2(n,n.loadP_t, n.loadP_t_agg, splitted_intervals, son_indeces_lists)
# %%
