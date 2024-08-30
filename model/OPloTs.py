
#%%
import numpy as np
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates
import os
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray # import_generated_scenario
from model.OPT_methods import OPT1, OPT2, OPT3, OPT_agg, OPT_time_partition
from model.EU_net import EU
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenarios
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# import plotly.graph_objects as go



def plotOPT3_secondstage(results, network, var, yaxis_title, title = "", xaxis_title = "Time"):
    # Initialize figure
    fig3 = go.Figure()
    if "node" in network.n.columns:
        network.n.set_index("node", inplace=True)

    if "node" in results[var].dims:
        nodes_or_edges = "nodes"
    else:
        nodes_or_edges = "edges"

    # Number of nodes and their names
    if nodes_or_edges == "nodes":
        N = network.n.shape[0]
        
        nodges = network.n.index.to_list()
        attr = 'n'
        dim = 'node'
    else:
        attr = 'edgesH'
        N = network.edgesH.shape[0]
        nodges = network.edgesH.set_index(["start_node","end_node"]).index.to_list()
        dim = 'edge'

    # Select the relevant data and assign new coordinates for time
    df = results[var]
    inst = len(network.genW_t.time)
    df = df.assign_coords({'time': range(inst)})
    # Summing over the dimension 'node'
    df_sum = df.sum(dim=dim)
    # Convert the summed DataArray to pandas DataFrame
    df_sum_df = df_sum.to_pandas().T
    # Add the summed line trace
    x = network.genW_t.time
    fig3.add_trace(go.Scatter(x=x, y=df_sum.values.flatten(), mode='lines', name='total'))

    # Adding individual node traces
    for i,n in enumerate(nodges):
        #print(n,i)
        if nodes_or_edges == "nodes":
            nodge_data = df.sel(scenario=0, node=n).values.flatten()
            name = n
        else:
            nodge_data = df.isel(scenario=0, edge=i).values.flatten()
            name = n[0] + " " + n[1]
        #print(f"Node: {n}, Data: {nodge_data}")
        fig3.add_trace(go.Scatter(
            x=x,
            y=nodge_data,
            mode='lines',
            name=name),
        )

    # Update the layout
    fig3.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Show the figure
    #fig3.show()
    return fig3
#%%

def node_results_df(results, network, vars = ["ns","nw","mhte"]):
    values = [results[var] for var in vars]
    var_dict = dict(zip(vars, values))
    nodes_pd = pd.DataFrame(var_dict)
    if "node" not in network.n.columns:
        network.n.reset_index(inplace=True)
    nodes_pd["node"] = network.n["node"]
    #nodes_pd.set_index("nodes", inplace=True)
    return nodes_pd
#%%
def plotOPT_pie(results,network, vars = ["ns","nw","mhte"],  label_names = ["Solar","Wind", "Power Cells"], title_text = "Percentual of maximum (MW) Power output"):
    if 'node' not in network.n.columns:
        network.n.reset_index(inplace=True)
    #"nh","meth",
  
    values = [results[var] for var in vars]
    var_dict = dict(zip(vars, values))
    nodes_pd = pd.DataFrame(var_dict)
    nodes_pd["nodes"] = network.n["node"]
    nodes_pd["MW wind"] = nodes_pd["nw"] * network.n["MP_wind"]
    nodes_pd["MW solar"] = nodes_pd["ns"] * network.n["MP_solar"]
    nodes_pd.set_index("nodes", inplace=True)

    if "mhte" in vars:
        nodes_pd["mhte"] = nodes_pd["mhte"] * 0.033*network.n["fhte"]

    # Create subplots with the correct type for pie charts
    fig = make_subplots(
        rows=2, 
        cols=len(nodes_pd.index), 
        subplot_titles=nodes_pd.index, 
        specs=[[{'type': 'domain'} for _ in range(len(nodes_pd.index))],[{'type': 'domain'} for _ in range(len(nodes_pd.index))]]
    )

    # Add pie charts to the subplots
    for i, n in enumerate(nodes_pd.index):
        fig.add_trace(go.Pie(values=nodes_pd[vars].loc[n], labels=vars, hole=0.3), row=1, col=i+1)
        fig.add_trace(go.Pie(values=nodes_pd[["MW wind", "MW solar"]].loc[n], labels=["MW wind", "MW solar"], hole=0.3), row=2, col=i+1)
   
    fig.update_layout(height=600, width=800, title_text=title_text)

    # Show the figure
    #fig.show()
    return fig
#%%
def plotE_balance(network, results, plot_H = True, x = None):
    title1 = 'Power output in each node for one scenario'
    xaxis_title1 = 'Time'
    yaxis_title1 = 'Power (MWh)'

    if "node" not in network.n.columns:
        network.n.reset_index(inplace=True)
    nodes = network.n['node']

    max_wind = xr.DataArray(network.n["MP_wind"], dims=['node'], coords = {'node': nodes})
    max_solar = xr.DataArray(network.n["MP_solar"], dims=['node'], coords = {'node': nodes})
    ns_da = xr.DataArray(results["ns"], dims = ['node'], coords = {'node': nodes})
    nw_da = xr.DataArray(results["nw"], dims = ['node'], coords = {'node': nodes})
    max_powercell = xr.DataArray(0.033*network.n["fhte"], dims=['node'], coords = {'node': nodes})
    ES = network.genS_t * ns_da
    EW = network.genW_t * nw_da
    
    EtH = results["EtH"]
    HtE = results["HtE"] * max_powercell
    #todo: plot load
    EL = network.loadP_t 
    HL = network.loadH_t
    #nodes = ['Italy','Spain', 'Italy', 'Spain']
    #electric production
    fig1 = make_subplots(
        rows=len(nodes), 
        cols=1, 
        subplot_titles=nodes, 
        specs=[[{'type': 'xy'}] for _ in range(len(nodes))]
    )
    if x is None:
        x = HtE.time
    # Add pie charts to the subplots
    for i, n in enumerate(nodes):
        if plot_H:
            y = HtE.sel(node = n, scenario = 0)
            fig1.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f"(MWh) Hydrogen fuelcell, {n}",
                    marker={'color' : 'red','opacity': 0.5}), row=i+1, col=1,
                    
            )

            y = EtH.sel(node = n, scenario = 0)
            fig1.add_trace(go.Scatter(
                    x=x,
                    y=-y,
                    mode='lines',
                    name=f"(MWh) Energy to Hydrogen, {n}",
                    marker={'color' : 'pink','opacity': 0.5}), row=i+1, col=1,
                    
            )
        y = ES.sel(node = n, scenario = 0)
        fig1.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f"(MWh) Solar, {n}",
                marker={'color':'lightblue'}), row=i+1, col=1,
            
            )

        y = EW.sel(node = n, scenario = 0)
        fig1.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f"(MWh) Wind, {n}",
                marker={'color' : 'blue'}), row=i+1, col=1
                
        )

        y = EL.sel(node = n, scenario = 0)
        fig1.add_trace(go.Scatter(
                x=x,
                y=-y,
                mode='lines',
                name=f"(MWh) Load, {n}",
                marker={'color' : 'orange'}), row=i+1, col=1,
                
        )
        



    fig1.update_layout(
            title=title1,
            xaxis_title=xaxis_title1,
            yaxis_title=yaxis_title1,
            height=1100,
            width=900,
            #title_text=title_text
        )

    return fig1

def plotH_balance(network, results, plot_H = True, x = None):
    title1 = 'Hydrogen balance in each node for one scenario'
    xaxis_title1 = 'Time'
    yaxis_title1 = 'Hydrogen (Kg)'

    if "node" not in network.n.columns:
        network.n.reset_index(inplace=True)
    nodes = network.n['node']

    feth = 30*xr.DataArray(network.n["feth"], dims=['node'], coords = {'node': nodes})
    EtH = results["EtH"] * feth
    HtE = results["HtE"]
    H = results["H"]
    title1 = 'Hydrogen Balance in each node for one scenario'
    xaxis_title1 = 'Time'
    yaxis_title1 = 'Hydrogen (Kg)'

    if "node" not in network.n.columns:
        network.n.reset_index(inplace=True)
    nodes = network.n['node']
    HtE = results["HtE"]
    HL = network.loadH_t
    fig2 = make_subplots(
        rows=len(nodes), 
        cols=1, 
        subplot_titles=nodes, 
        specs=[[{'type': 'xy'}] for _ in range(len(nodes))]
    )
    if x is None:
        x = H.time
    # Add pie charts to the subplots
    for i, n in enumerate(nodes):
        y = HL.sel(node = n, scenario = 0)
        fig2.add_trace(go.Scatter(
                x=x,
                y= -y,
                mode='lines',
                name=f"(Kg) Hydrogen Load, {n}",
                marker={'color':'orange'}), row=i+1, col=1,
            
            )
        y = HtE.sel(node = n, scenario = 0)
        fig2.add_trace(go.Scatter(
                x=x,
                y= -y,
                mode='lines',
                name=f"(Kg) Hydrogen to Energy, {n}",
                marker={'color' : 'red'}), row=i+1, col=1   
        )

        y = EtH.sel(node = n, scenario = 0)
        fig2.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f"(Kg) Energy to Hydrogen, {n}",
                marker={'color' : 'blue'}), row=i+1, col=1   
        )

        y = H.sel(node = n, scenario = 0)
        fig2.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f"(Kg) Hydrogen Storage, {n}",
                marker={'color' : 'green'}), row=i+1, col=1,
                
        )

    fig2.update_layout(
            title=title1,
            xaxis_title=xaxis_title1,
            yaxis_title=yaxis_title1,
            height=1100,
            width=900,
            #title_text=title_text
        )


    return fig2

#%%
def plotOPT_time_partition(results, network,var, title, yaxis_title):
    # results = results[-1] result dict
    # Initialize figure
    fig = go.Figure()
    if "node" in network.n.columns:
        network.n.set_index("node", inplace=True)
    # Number of nodes and their names
    Nnodes = network.n.shape[0]
    nodes = network.n.index.to_list()
    # Select the relevant data and assign new coordinates for time
    df = results[var]
    tp = network.time_partition.agg
    interval_to_var = results['interval_to_var']
    last_vars = [interval_to_var[t] for t in network.time_partition.tuplize(tp)]
    df = df.sel(time=last_vars)  # restrict to last tp variables
    df = df.assign_coords({'time': range(len(tp))})

    # Print statements to debug the data
    # print("Original DataFrame:\n", df)
    # print("Time coordinates:\n", df.coords['time'].values)
    # print("Nodes:\n", df.coords['node'].values)

    # Summing over the dimension 'node'
    df_sum = df.sum(dim='node')
    #print("Summed DataFrame:\n", df_sum)

    # Convert the summed DataArray to pandas DataFrame
    df_sum_df = df_sum.to_pandas().T
    #print("Summed DataFrame (Pandas):\n", df_sum_df)

    # Add the summed line trace
    fig.add_trace(go.Scatter(x=df_sum.coords['time'].values, y=df_sum.values.flatten(), mode='lines', name='total'))

    # Adding individual node traces
    for n in nodes:
        node_data = df.sel(scenario=0, node=n).values.flatten()
        #print(f"Node: {n}, Data: {node_data}")
        fig.add_trace(go.Scatter(
            x=df.coords['time'].values,
            y=node_data,
            mode='lines',
            name=n),
        )

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=yaxis_title,
        height=700,
        width=800,
    )

    # Show the figure
    #fig.show()
    return fig
# %%
