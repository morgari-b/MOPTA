


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
#TODO do not redefine EL, HL and so on for every iteration.

eu1 = EU(10, init_method= "day_night_aggregation") #,init_method = 'day_night_aggregation'
eu2 = EU(10, init_method= "day_aggregation")
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


network = eu1
N_iter = 3
# def OPT_agg2(network):
#     """
#     Same as OPT_agg but adds all T variables from the start
#     (this way we have a constraint generation for all T variables)
#     """
if network.costs.shape[0] == 1: #if the costs are the same:
    cs, cw, ch, ch_t, chte, ceth, cNTC, cMH = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0]
else:
    print("add else") #actually we can define the costs appropriately using the network class directly


start_time=time.time()
Nnodes = network.n.shape[0]
NEedges = network.edgesP.shape[0]
NHedges = network.edgesH.shape[0]
d = network.n_scenarios 
T = network.T 
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


ES = network.genS_t
EW = network.genW_t
EL = network.loadP_t
HL = network.loadH_t


if network.loadP_t_agg.shape[2] > 1:
    cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                    quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                    quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                    == HL.isel(node=k, time=tp[i], scenario = j).sum(dim = 'time') for j in range(d) for i in range(Ntp) for k in range(Nnodes)))
    
    cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                        quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                        >= EL.isel(node=k, time=tp[i], scenario = j).sum(dim = 'time') for k in range(Nnodes) for j in range(d) for i in range(Ntp)))

else:
    cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                    quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                    quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                    == HL.isel(node=k, time=tp[i], scenario = 0).sum(dim = 'time') for j in range(d) for i in range(Ntp) for k in range(Nnodes) )) #

    cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j]  + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                        quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                        >= EL.isel(node=k, time=tp[i], scenario = 0).sum(dim = 'time') for k in range(Nnodes)  for j in range(d) for i in range(Ntp)))
print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
opt_start_time = time.time()
model.optimize()
iter_sol = []
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
    iter_sol.append(VARS)
    print(f"opt time {np.round(time.time()-opt_start_time,3)}s.")
                                            

print("total time: {}s.".format(np.round(time.time()-start_time,3)))
#%%    return VARS
for iter in range(10):
    iter_start_time = time.time()
    network.time_partition.random_iter_partition(k=1)
    family_tree = network.time_partition.family_tree
    splitted_intervals = time_partition.order_intervals(family_tree[-1])
    #print(splitted_intervals)
    tp_obj = network.time_partition #new time partition object
    tp = tp_obj.agg.copy()
    tp = [t if type(t) is list else [t] for t in tp]


    for father_interval in splitted_intervals:
        split_indeces = time_partition.interval_subsets(father_interval,tp)[1:] #indexes of tp that are subsets of the father_interval, except the first one sine it's implied by linear dependence of the corresponding contraints.
    

        if network.loadP_t_agg.shape[2] > 1:
            cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                            quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            == HL.isel(node=k, time=tp[i], scenario = j).sum(dim = 'time') for j in range(d) for i in split_indeces for k in range(Nnodes)))
            
            cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                                quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                >= EL.isel(node=k, time=tp[i], scenario = j).sum(dim = 'time') for k in range(Nnodes) for j in range(d) for i in split_indeces))

        else:
            cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                            quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            == HL.isel(node=k, time=tp[i], scenario = 0).sum(dim = 'time') for j in range(d) for i in split_indeces for k in range(Nnodes) )) #

            cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j]  + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                                quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                >= EL.isel(node=k, time=tp[i], scenario = 0).sum(dim = 'time') for k in range(Nnodes)  for j in range(d) for i in split_indeces))
    print(f"Iter model time: {np.round(time.time()-iter_start_time,3)}s.")
    model.optimize()
    print(f"Iter opt time: {np.round(time.time()-iter_start_time,3)}s.")
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
        iter_sol.append(VARS)
    

print(f"Total opt time: {np.round(time.time()-opt_start_time,3)}s.")
    

# %%
