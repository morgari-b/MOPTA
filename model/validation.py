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

#%%

#  VALIDATION FUNCTION

# ideas: 
#   - get average scenario, optimize on that
#   - loss function to optimal scenario
#   - rolling horizon optimization 2days at a time
#   - messages telling me when it fails

#TODO: capire perchè è feasible anche se non usa stoccaggio di idrogeno
#TODO: sistemare relazione cona ggregazione temporale (così) i timestep della domanda sono sfasati con lo stoccaggio di idrogeno
#TODO: dividere costi di trasmissione per numero di scenari.

def Validate(network,VARS,scenarios):
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    goalH=VARS[0]['H'].mean('scenario').to_pandas()
    t=goalH.shape[0]
    goalH.loc[t,:]=goalH.loc[0,:]
    goalH=goalH.set_index(pd.date_range("2023-01-01", "2024-01-01, 00:00:00",periods=t+1)).resample("h").interpolate("linear").head(-1)
    
    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    #d = network.n_scenarios 
    d = scenarios['wind_scenario'].shape[2]
    #inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    inst = 24
    #tp_obj = network.time_partition
        #tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    #ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    #nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    #nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    #mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    #meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    #addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    #addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k]-H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k]-H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss,lb=-GRB.INFINITY)
    
    # starting hydrogen levels
    values = np.zeros([d,1,len(network.n.index.to_list())])
    Hs = xr.DataArray(values, 
                         dims=["scenario","time","node"], 
                         coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range('01 jan,2023 00:00:00',freq='h',periods=1),network.n.index.to_list()])))
    #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
    Hs[:,0,:]=VARS[0]['H'][:,0,:].max('scenario') # else might get unfeasibility for net 0 solutions
    
    
    c1=model.addConstr(H[0,0,0]>=0)
    c2=model.addConstr(H[0,0,0]>=0)
    c3=model.addConstr(H[0,0,0]>=0)
    c4=model.addConstr(H[0,0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        
        day_num = 0
        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[:,24*day_num:24*(day_num+1),:]
        
        #MAKE THE INDICES MAKE SENSE
        day_num=day_num+1
    
        model.remove(c1)
        model.remove(c2)
        model.remove(c3)
        model.remove(c4)
        #model.remove(c5)
        
        # if network.loadP_t_agg.shape[2] > 1:
        #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
    
            
        #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
        
        #else:
    
            
        # changed index compared to OPT3: H[i] is the storage at the end of hour i.
        c1 = model.addConstrs((- H[j,i,k] + H[j,i-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(1,inst) for k in range(Nnodes)))
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("failed at " + str(day))
            break
        else:
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return Hs
    
    
    
            
        
    


# %% trials

eu=EU()
VARS=OPT_agg_correct(eu)

#%%
scenarios = import_scenario_val(5,5)
#%%
Validate(eu,VARS,scenarios)

#%%
eu1 = EU(n_scenarios=1)
VARS1 = OPT3(eu1)

#%%
Hs = Validate(eu1,VARS1,scenarios)


# %% debugging
