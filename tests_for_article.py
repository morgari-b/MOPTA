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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def opt3_size(n_nodes: int,
              n_elec_edges: int,
              n_h2_edges: int,
              n_time: int,
              n_scen: int,
              pretty: bool = False):
    """
    Return (n_variables, n_constraints) for the exact model `OPT3`
    as written in the question.

    Parameters
    ----------
    n_nodes : int          # |𝒩|
    n_elec_edges : int     # |ℰᴾ|
    n_h2_edges : int       # |ℰᴴ|
    n_time : int           # |𝒯|
    n_scen : int           # |𝒮|
    pretty : bool, default False
        If True, print a short report before returning.

    Returns
    -------
    tuple[int, int] : (V, C)
    """
    # --- variables ----------------------------------------------------------
    v_static_nodes  = 5 * n_nodes
    v_static_edges  = n_elec_edges + n_h2_edges
    v_dyn_nodes     = 3 * n_nodes * n_time * n_scen
    v_dyn_elec      = 2 * n_elec_edges * n_time * n_scen
    v_dyn_h2        = 2 * n_h2_edges * n_time * n_scen
    n_vars = (v_static_nodes + v_static_edges +
              v_dyn_nodes + v_dyn_elec + v_dyn_h2)

    # --- constraints --------------------------------------------------------
    c_nodes   = 5 * n_nodes * n_time * n_scen
    c_elec    = 4 * n_elec_edges * n_time * n_scen
    c_h2      = 4 * n_h2_edges * n_time * n_scen
    n_cons = c_nodes + c_elec + c_h2

    if pretty:
        print(f"OPT3 size for |𝒩|={n_nodes}, |ℰᴾ|={n_elec_edges}, "
              f"|ℰᴴ|={n_h2_edges}, |𝒯|={n_time}, |𝒮|={n_scen}")
        print(f"  variables   : {n_vars:_}")
        print(f"  constraints : {n_cons:_}")

    return n_vars, n_cons

def opt_size(network, n_scen = 'Nan', n_time = 'Nan'):
    n_nodes = network.n_nodes()
    n_elec_edges = network.n_elec_edges()
    n_h2_edges = network.n_h2_edges()
    if n_scen == 'Nan':
        n_scen = network.n_scen()
    else:
        n_scen = n_scen
    if n_time == 'Nan':
        n_time = network.n_time()
    else:
        n_time = n_time
    return opt3_size(n_nodes, n_elec_edges, n_h2_edges, n_time, n_scen)
def opt_agg_iter(network_input, N_iter, iter_method, k, max_time):
    """
    Run the optimization algorithm with the specified parameters.

    Args:
        network (Network): The network object to optimize.
        N_iter (int): The number of iterations.
        iter_method (str): The method for iteration ('rho' or 'random').
        k (int): The number of disaggregations per iteration.
        stop_condition (str): The stopping condition ('max_time' or 'max_iter').
        max_time (int): The maximum time allowed for the optimization.

    Returns:
        list: A list of dictionaries containing the results of each iteration.
    """
    network = copy.deepcopy(network_input)
    total_opt_times = []
    total_opt_time = time.time()
    iter_sol = []
    for i in range(N_iter):
        logging.info(f"Iteration {i+1}/{N_iter}")
        start_opt_time = time.time()
        VARS = OPT_agg(network)
        end_opt_time = time.time()
        last_opt_time = total_opt_time
        total_opt_time = end_opt_time - start_opt_time
        if total_opt_time > max_time:
            total_opt_times.append(total_opt_time)
            iter_sol = iter_sol + [VARS]
            VARS[0]['total_opt_time'] = total_opt_time
            VARS[0]['iter_opt_time'] = total_opt_time - last_opt_time
            break
        total_opt_times.append(total_opt_time)
        iter_sol = iter_sol + [VARS]
        if iter_method == "random":
            print("random iteration")
            network.iter_partition(k=k)
        elif iter_method == "total":
            print("total iteration")
            network.total_iter_partition()  
        elif iter_method == "rho":
            print("rho iteration")
            network.rho_iter_partition(VARS[0], k=k)
        else:
            raise ValueError(f"Unknown iteration method: {iter_method}")
    return iter_sol

def cost_and_times(vars):
    # Extract obj, total_opt_time, and iter_opt_time from each dictionary in vars
    objs = [entry['obj'] for entry in vars]
    total_opt_times = [entry['total_opt_time'] for entry in vars]
    iter_opt_times = [entry['iter_opt_time'] for entry in vars]
    return objs, total_opt_times, iter_opt_times

def save_results_to_json(opt_name, costs, total_opt_times, iter_opt_times):
    print(f'starting saving for {case,opt_name}')
    os.makedirs("tests", exist_ok=True)
    with open(f"tests/result_{case}_opt{opt_name}_iter{N_iter}_random{N_random_tests}_time{max_time}.json", "w") as f:
        json.dump({
        "costs": costs,
        "total_opt_times": total_opt_times,
        "iter_opt_times": iter_opt_times_rho,
    }, f, indent=4)
# %% set parameters
cases = ['case5','case9','case14','case30','case57','case118','case300'] 
max_time = 1700
n_disaggregations_per_iterarion = 3
N_random_tests = 1
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
complexity_dict = {}
# Loop through each case and call the corresponding function
for case in cases:

    
    print(f'case {case}')
    # Dynamically call the function using getattr
    case_function = getattr(pn, case)  # Get the function from the pn module
    network = case_function()  # Call the function
    print(f"Loaded network for {case}:")
    print(network)
    logging.info(f"Converting network for {case}:")
    network = convert_pandapower_to_yuppy(network)
    logging.info(f"Converted network for {case}:")
    network = assign_random_scenarios_to_nodes(network, 1)
    logging.info(f"Assigned random scenarios to nodes for {case}:")
    network.plot()
    #N_iter = int(np.ceil(network.Tagg/ n_disaggregations_per_iterarion))

    #complexity
    #complexity_dict[case] = opt_size(network, n_scen = 10, n_time = 24*365)
    # iter agg
    # logging.info(f"Starting aggregation iteration for {case}:")
    # res_agg_iter_rho = opt_agg_iter(network, N_iter = 100,  iter_method = "rho", k = n_disaggregations_per_iterarion, max_time = max_time)
    # costs_agg_iter_rho, total_opt_times_agg_iter_rho, iter_opt_times_agg_iter_rho= cost_and_times(res_agg_iter_rho)
    # save_results_to_json('agg_iter_rho', costs_agg_iter_rho, total_opt_times_agg_iter_rho, iter_opt_times_agg_iter_rho)
    # # rho test
    print(f'starting rho iter for {case}')
    res_iter_rho = OPT_agg2(network, N_iter = N_iter,  iter_method = "rho", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
    costs_rho, total_opt_times_rho, iter_opt_times_rho= cost_and_times(res_iter_rho)
    save_results_to_json('rho', costs_rho, total_opt_times_rho, iter_opt_times_rho)
    
    # vars_random_list = []
    # costs_random_list = []
    for i in range(N_random_tests):
        print(f'starting random iter number {i} for {case}')
        print(f"iteration {i+1}")
        vars_random =  OPT_agg2(network, N_iter = N_iter,  iter_method = "random", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
        vars_random_list.append(vars_random)
        # costs_random = [vars_random[i]['obj'] for i in range(len(vars_random))]
        # costs_random_list.append(costs_random)
        costs, total_opt_times_random, iter_opt_times_random= cost_and_times(vars_random)
        save_results_to_json(f'random_{i}', costs, total_opt_times_random, iter_opt_times_random)
    #
    # total_opt_times_random_list = []
    # iter_opt_times_random_list = []
    # for i in range(N_random_tests):
    #     total_opt_times_random = [vars_random_list[i][j]['total_opt_time'] for j in range(len(vars_random_list[i]))]
    #     iter_opt_times_random = [vars_random_list[i][j]['iter_opt_time'] for j in range(len(vars_random_list[i]))]
    #     total_opt_times_random_list.append(total_opt_times_random)
    #     iter_opt_times_random_list.append(iter_opt_times_random)

    # #
    # #run unaggregated
    # n = copy.deepcopy(network)
    # n.total_iter_partition()
    # print(f'starting exact solver for {case}')
    # res_exact = OPT_agg2(n, N_iter = 0,  iter_method = "random", k = n_disaggregations_per_iterarion, stop_condition = "max_time", max_time = max_time)
    # costs_exact, total_opt_times_exact, iter_opt_times_exact= cost_and_times(res_exact)
    # save_results_to_json('exact', costs_exact, total_opt_times_exact, iter_opt_times_exact)


  
    # save results
    # Ensure the "tests" folder exists



#%%

file_name = f"tests/results_{case}_iter{N_iter}_random{N_random_tests}_timemax{max_time}.json"
try:
    with open(file_name, "r") as f:
        results = json.load(f)
        print(f"Results for {case}:")
        print(results)
except FileNotFoundError:
    print(f"File {file_name} not found.")

# %%
