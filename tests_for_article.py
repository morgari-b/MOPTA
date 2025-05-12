# %% Import packages
import json
import copy
import pandapower.networks as pn
import numpy as np
import random
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates
import os
from model.network import  Network # import_generated_scenario
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario
from model.OPT_methods import OPT_agg, OPT_agg2, OPT_time_partition, OPT3
from model.create_istances import convert_pandapower_to_yuppy, assign_random_scenarios_to_nodes, EU
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cost_and_times(vars):
    # Extract obj, total_opt_time, and iter_opt_time from each dictionary in vars
    objs = [entry['obj'] for entry in vars]
    total_opt_times = [entry['total_opt_time'] for entry in vars]
    iter_opt_times = [entry['iter_opt_time'] for entry in vars]
    return objs, total_opt_times, iter_opt_times
# %% set parameters
cases = ['case5','case118','case300','case1354pegase','case2869pegase','case9241pegase',]
max_time = 1200
n_disaggregations_per_iterarion = 3
N_random_tests = 5
N_iter = 100000

# network = convert_pandapower_to_yuppy(network)
# network = assign_random_scenarios_to_nodes(network, 1)
# network.plot()
# N_iter = int(np.ceil(network.Tagg/ n_disaggregations_per_iterarion))

#%% run unaggregated
# case_function = getattr(pn, 'case30')  # Get the function from the pn module
# network = case_function()  # Call the function
# network = convert_pandapower_to_yuppy(network)
# network = assign_random_scenarios_to_nodes(network, 1)
# network.plot()
# print(network)
# n = copy.deepcopy(network)
# n.total_iter_partition()
# res_exact = OPT_agg2(n, N_iter = 0,  iter_method = "random", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
    #
#%%
# Loop through each case and call the corresponding function
for case in cases:
    print(f'case {case}')
    # Dynamically call the function using getattr
    case_function = getattr(pn, case)  # Get the function from the pn module
    network = case_function()  # Call the function
    print(f"Loaded network for {case}:")
    print(network)

    network = convert_pandapower_to_yuppy(network)
    network = assign_random_scenarios_to_nodes(network, 1)
    network.plot()
    N_iter = int(np.ceil(network.Tagg/ n_disaggregations_per_iterarion))
    
    # run unaggregated
    # n = copy.deepcopy(network)
    # n.total_iter_partition()
    # print(f'starting exact solver for {case}')
    # res_exact = OPT_agg2(n, N_iter = 0,  iter_method = "random", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
    #
    # rho test
    print(f'starting rho iter for {case}')
    res_iter_rho = OPT_agg2(network, N_iter = N_iter,  iter_method = "rho", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)

    # random test: need to take average over multiple runs
    
    vars_random_list = []
    costs_random_list = []
    for i in range(N_random_tests):
        print(f'starting random iter number {i} for {case}')
        print(f"iteration {i+1}")
        vars_random =  OPT_agg2(network, N_iter = N_iter,  iter_method = "random", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
        vars_random_list.append(vars_random)
        costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
        costs_random_list.append(costs_random)
    #
    total_opt_times_random_list = []
    iter_opt_times_random_list = []
    for i in range(N_random_tests):
        total_opt_times_random = [vars_random_list[i][j]['total_opt_time'] for j in range(len(vars_random_list[i]))]
        iter_opt_times_random = [vars_random_list[i][j]['iter_opt_time'] for j in range(len(vars_random_list[i]))]
        total_opt_times_random_list.append(total_opt_times_random)
        iter_opt_times_random_list.append(iter_opt_times_random)

   

    costs_rho, total_opt_times_rho, iter_opt_times_rho= cost_and_times(res_iter_rho)
    #costs_exact, total_opt_times_exact, iter_opt_times_exact= cost_and_times(res_exact)
    # save results
    # Ensure the "tests" folder exists
    print(f'starting saving for {case}')
    os.makedirs("tests", exist_ok=True)
    with open(f"tests/result_{case}_iter{N_iter}_random{N_random_tests}_time{max_time}.json", "w") as f:
        json.dump({
            "costs_rho": costs_rho,
            "total_opt_times_rho": total_opt_times_rho,
            "iter_opt_times_rho": iter_opt_times_rho,
            #"costs_exact": costs_exact,
            #"total_opt_times_exact": total_opt_times_exact,
            #"iter_opt_times_exact": iter_opt_times_exact,
            "costs_random_list": costs_random_list,
            "total_opt_times_random_list": total_opt_times_random_list,
            "iter_opt_times_random_list": iter_opt_times_random_list,
        }, f, indent=4)


#%%
# Loop through each case and load the corresponding results
for case in cases:
    file_name = f"tests/results_{case}_iter{N_iter}_random{N_random_tests}_timemax{max_time}.json"
    try:
        with open(file_name, "r") as f:
            results = json.load(f)
            print(f"Results for {case}:")
            print(results)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

# %%
