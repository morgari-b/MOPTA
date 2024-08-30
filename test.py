


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
os.chdir("C:/Users/ghjub/codes/MOPTA")
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray # import_generated_scenario
from model.OPT_methods import OPT1, OPT2, OPT3, OPT_agg, OPT_time_partition, OPT_time_partition_old
from model.EU_net import EU
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenarios, import_scenario
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model.OPloTs import node_results_df, plotOPT3_secondstage, plotOPT_time_partition, plotOPT_pie, plotE_balance, plotH_balance


#%%

eu1 = EU(2, init_method= "day_night_aggregation") #,init_method = 'day_night_aggregation'
eu2 = EU(2, init_method= "day_aggregation")
#%%
eu = EU(1)
#%%
results1 = OPT_agg(eu1)
#%%
results2 = OPT_agg(eu2)
#%%
results3 = OPT3(eu)
#%%
results4 = OPT_time_partition(eu, N_iter = 5, N_refining = 1)
#results = OPT_time_partition(eu)

#%%
results = results3
network = eu
hydro_fig = plotOPT3_secondstage(results, network, "H",  yaxis_title = "Hydrogen Storage (Kg)", title = "Hydrogen Storage over time")
P_edge_fig = plotOPT3_secondstage(results, network, "P_edge", yaxis_title="Power flow (MWh)", title = "Power flow through lines", xaxis_title = "Time")
#node_results = node_results_df(results)
first_text = "Hover over the graph to see the explicit optimization results"
generators_pie_fig =  plotOPT_pie(results,network, vars = ["ns","nw","mhte"],  label_names = ["Solar","Wind", "Power Cells"], title_text = "Number of generator by type and percentage of maximum energy production")
x = network.genS_t.time
Ebalance_fig = plotE_balance( network, results, x = x)
Hbalance_fig = plotH_balance(network, results, x= x)

#%%
fig1 = plotOPT_time_partition(results1[-1], eu1, 'H', title = "Hydrogen storage over time", yaxis_title="Hydronge storage (kg)")
fig1.show()
fig2 = plotOPT_time_partition(results2[-1], eu2, 'H', title = "Hydrogen storage over time", yaxis_title="Hydronge storage (kg)")
fig2.show()
#%%
fig = plotOPT3_secondstage(results3, eu, 'H', yaxis_title = "Hydrogen storage (Kg)", title = "Hydrogen storage over time", xaxis_title = "Time")
fig.show()
# plotOPT3_secondstage, plotOPT_time_partition, plotOPT_pie, plotE_balance, plotH_balance

#%%

# network = eu
# last_results = results[-1]
# generators_pie_fig =  plotOPT_pie(last_results,network, vars = ["ns","nw","mhte"],  label_names = ["Solar","Wind", "Power Cells"], title_text = "Percentual of maximum (KW) Power output")
# generators_pie_fig.show()
# plotE_balance_fig = plotE_balance( network, last_results)
# plotE_balance_fig.show()
# plotH_balance_fig = plotH_balance( network, last_results)
# plotH_balance_fig.show()

# #results3 = OPT3(eu)

#%%

 """
    Initializes a network object for the European Union.

    This function creates a network object for the European Union by initializing the necessary attributes. It sets the index of the 'EU' DataFrame to 'node' and assigns values to the 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh' columns. It also creates two DataFrames, 'EU_e' and 'EU_h', with columns 'start_node' and 'end_node'. The 'EU_e' and 'EU_h' DataFrames are assigned values to the 'NTC' and 'MH' columns respectively. Finally, it creates a 'costs' DataFrame with columns 'node', 'cs', 'cw', 'ch', 'chte', 'ceth', 'cNTC', and 'cMH'.

    Returns:
        None
    """

n_scenarios = 3
init_method = "day_night_aggregation"   




#%%
#results = results3
#network = eu
#def plotOPT_energy_generation(results, network):

#%%
#results_agg = OPT_agg(eu)
# 1- select only variable scorresponding to last tp
# 2- expand solutions
# for energy production we can simply use original unaggregated scenarios.
# for hydrogen we could plot cose a scala (e eventualmente interpolare)
# for p_edge prendere la media e so on
#TODO: plot objective functions over iterations.
#TODO: 
# def plot_results(results):


#%%
print("miao")

# %% validaiton 
network = eu
res = results1[-1]



def validation_function(res, network, scen):
        if network.costs.shape[0] == 1:  # if the costs are the same:
            cs, cw, ch, ch_t, chte, ceth, cNTC, cMH = (network.costs['cs'][0], network.costs['cw'][0],
                                                    network.costs['ch'][0], network.costs['ch_t'][0],
                                                    network.costs['chte'][0], network.costs['ceth'][0],
                                                    network.costs['cNTC'][0], network.costs['cMH'][0])
        else:
            print("add else")  # actually we can define the costs appropriately using the network class directly

        if "node" in network.n.columns:
            network.n.set_index("node", inplace=True)

        start_time = time.time()
        Nnodes = network.n.shape[0]
        NEedges = network.edgesP.shape[0]
        NHedges = network.edgesH.shape[0]
        d_loadP = network.loadP_t.shape[2]  # number of scenarios for demand
        inst = network.loadP_t.shape[0]  # number of time steps T
        env = Env(params={'OutputFlag': 0})
        model = Model(env=env)
        model.setParam('LPWarmStart', 1)

        ns = res['ns']
        nw = res['nw']
        print("N solar, N wind:",ns,nw)
        nh = res['nh']
        mhte = res['mhte']
        meth = res['meth']
        addNTC = res['addNTC']
        addMH = res['addMH']

        HtE = model.addVars(product(range(d), range(inst), range(Nnodes)), vtype=GRB.CONTINUOUS, lb=0)  # expressed in kg      
        EtH = model.addVars(product(range(d), range(inst), range(Nnodes)), vtype=GRB.CONTINUOUS, lb=0)  # expressed in MWh
        H = model.addVars(product(range(d), range(inst), range(Nnodes)), vtype=GRB.CONTINUOUS, lb=0)
        P_edge = model.addVars(product(range(d), range(inst), range(NEedges)), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        H_edge = model.addVars(product(range(d), range(inst), range(NHedges)), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)

        model.addConstrs(H[j, i, k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
        model.addConstrs(EtH[j, i, k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
        model.addConstrs(HtE[j, i, k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
        model.addConstrs(P_edge[j, i, k] <= network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
        model.addConstrs(H_edge[j, i, k] <= network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))
        model.addConstrs(-P_edge[j, i, k] <= network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
        model.addConstrs(-H_edge[j, i, k] <= network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))

       

        ES = network.genS_t
        EW = network.genW_t
        EL = network.loadP_t
        HL = network.loadH_t

        if d_loadP == 1:
            print("Adding E and H balance constraints")
            cons2 = model.addConstrs(-H[j, (i + 1) % inst, k] + H[j, i, k] + 30 * network.n['feth'].iloc[k] * EtH[j, i, k] - HtE[j, i, k] -
                                    quicksum(H_edge[j, i, l] for l in network.edgesH.loc[network.edgesH['start_node'] == network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(H_edge[j, i, l] for l in network.edgesH.loc[network.edgesH['end_node'] == network.n.index.to_list()[k]].index.to_list())
                                    == HL[i, k, 0] for j in range(d) for i in range(inst) for k in range(Nnodes))
            cons1 = model.addConstrs(0.033 * network.n['fhte'].iloc[k] * HtE[j, i, k] - EtH[j, i, k] -
                                    quicksum(P_edge[j, i, l] for l in network.edgesP.loc[network.edgesP['start_node'] == network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(P_edge[j, i, l] for l in network.edgesP.loc[network.edgesP['end_node'] == network.n.index.to_list()[k]].index.to_list())
                                    >= EL[i, k, 0] - ns[k] * ES[i, k, scen] - nw[k] * EW[i, k, scen] for k in range(Nnodes) for j in range(d) for i in range(inst))
        else:
            cons2 = model.addConstrs(-H[j, (i + 1) % inst, k] + H[j, i, k] + 30 * network.n['feth'].iloc[k] * EtH[j, i, k] - HtE[j, i, k] -
                                    quicksum(H_edge[j, i, l] for l in network.edgesH.loc[network.edgesH['start_node'] == network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(H_edge[j, i, l] for l in network.edgesH.loc[network.edgesH['end_node'] == network.n.index.to_list()[k]].index.to_list())
                                    == HL[i, k, j] for j in range(d) for i in range(inst) for k in range(Nnodes))
            cons1 = model.addConstrs(0.033 * network.n['fhte'].iloc[k] * HtE[j, i, k] - EtH[j, i, k] -
                                    quicksum(P_edge[j, i, l] for l in network.edgesP.loc[network.edgesP['start_node'] == network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(P_edge[j, i, l] for l in network.edgesP.loc[network.edgesP['end_node'] == network.n.index.to_list()[k]].index.to_list())
                                    >= EL[i, k, scen] - ns[k] * ES[i, k, scen] - nw[k] * EW[i, k, scen] for k in range(Nnodes) for j in range(d) for i in range(inst))
        print('OPT Model has been set up, this took ', np.round(time.time() - start_time, 4), 's.')
        model.optimize()

        if model.Status != 2:
            print("Unfeasible or Unbounded Status = {}".format(model.Status))
            model.computeIIS()
            constrs = model.getConstrs()
            IIS = []
            for c in constrs:
                if c.IISConstr:
                    IIS.append(c)
            print(IIS)
            print("returning IIS")
            return IIS
        else:
            print("Solution is feasible for current scenario.")
            print("opt time: {}s.".format(np.round(time.time() - start_time, 3)))


#%%


network = eu
def OPT_agg2(network):
    """
    Same as OPT_agg but adds all T variables from the start
    (this way we have a constraint generation for all T variables)
    """
    if network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, ch_t, chte, ceth, cNTC, cMH = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly


    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = network.n_scenarios 
    T = eu.T 
    tp_obj = network.time_partition
    tp = []
    for t in tp_obj.agg:
        if type(t) is int:
            tp += [[t]]
        elif len(t) > 0:
            tp += [t]
    #time partition
    Ntp = len(tp)

    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)

    HtE = model.addVars(product(range(d),range(T),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(T),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(T),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(range(d),range(T),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(T),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(T) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(T) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(T) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(T) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(T) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(T) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(T) for j in range(d) for k in range(NHedges))

    outputs=[]
    VARS=[]


    ES = network.genS_t_agg
    EW = network.genW_t_agg
    EL = network.loadP_t_agg
    HL = network.loadH_t_agg


    if network.loadP_t_agg.shape[2] > 1:
        cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,j] for j in range(d) for i in range(Ntp) for k in range(Nnodes)))
        
        cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(Ntp)))

    else:
        cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(Ntp) for k in range(Nnodes) )) #

        cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j]  + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes)  for j in range(d) for i in range(Ntp)))
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    opt_start_time = time.time()
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        node_dims = ["scenario","time","node"]
        if 'node' in network.n.columns:
            network.n.set_index('node',inplace=True)
        node_coords = [ range(d), range(T),  network.n.index.to_list()]
        edge_dims = ["scenario","time","edge"]
        edge_coords = [ range(d),  range(T), range(NEedges)]
        VARS={
            "ns":np.ceil([ns[k].X for k in range(Nnodes)]),
            "nw":np.ceil([nw[k].X for k in range(Nnodes)]),
            "nh":np.array([nh[k].X for k in range(Nnodes)]),
            "mhte":np.array([mhte[k].X for k in range(Nnodes)]),
            "meth":np.array([meth[k].X for k in range(Nnodes)]),
            "addNTC":np.array([addNTC[l].X for l in range(NEedges)]),
            "addMH":np.array([addMH[l].X for l in range(NHedges)]),
            "H":solution_to_xarray(H, node_dims, node_coords),
            "EtH":solution_to_xarray(EtH, node_dims, node_coords),
            "P_edge":solution_to_xarray(P_edge, edge_dims, edge_coords),
            "H_edge":solution_to_xarray(H_edge, edge_dims, edge_coords),
            "HtE":solution_to_xarray(HtE, node_dims, node_coords),
            "obj":model.ObjVal,
            "interval_to_var":dict(zip(time_partition.tuplize(tp),range(T))),
            "var_to_interval":dict(zip(range(T),time_partition.tuplize(tp)))  
        }
        print(f"opt time {np.round(time.time()-opt_start_time,3)}s.")
                                                    

    print("total time: {}s.".format(np.round(time.time()-start_time,3)))
    return VARS
# %%
