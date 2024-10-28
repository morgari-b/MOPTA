#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:05:49 2024v
@author: morgari
CURRENT WORKING MODELS:
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates
import os
#os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")
import xarray as xr
import folium
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario, scenario_to_array
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#%%

#%%

#  VALIDATION FUNCTION

# ideas: 
#   - get average scenario, optimize on that
#   - loss function to optimal scenario
#   - rolling horizon optimization 2days at a time
#   - messages telling me when it fails

#TODO: capire perchè è feasible anche se non usa stoccaggio di idrogeno
#TODO: AAA sistemare relazione cona ggregazione temporale (così) i timestep della domanda sono sfasati con lo stoccaggio di idrogeno


#%% old Validate
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
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
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
    
   
    day_num = 0
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
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
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            break
        else:
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                #print(key)
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return day_num
def Validate3(network,VARS,scenarios):
         
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
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
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
    
   
    day_num = 0
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
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
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            return day_num
        else:
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                #print(key)
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return day_num


def Validate2(network,VARS,scenarios, day_initial, scenario_initial):
    """
    Like validate but starting from time t_initial and scenario_initial and doing one scenario at the time.
    """
         
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
    
    HtE = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[i,k] <= nh[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( EtH[i,k] <= meth[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( HtE[i,k] <= mhte[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
    if 'node' in network.n.columns:
        network.n = network.n.set_index('node')

    c1=model.addConstr(H[0,0]>=0)
    c2=model.addConstr(H[0,0]>=0)
    c3=model.addConstr(H[0,0]>=0)
    c4=model.addConstr(H[0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    T = len(pv_scenario.time)
    n_days = int(np.floor(T/24))
    H_list = []
    for j in range(scenario_initial,d):
        values = np.zeros([1,len(network.n.index.to_list())])
        if j == scenario_initial: #we start from day day_initial and scenario scenario_initial and then continue as normal
            day_num = day_initial
             # starting hydrogen level
            tp = network.time_partition.agg
            for i in range(len(tp)):
                if type(tp[i]) is list:
                    if day_num*24 in tp[i]:
                        day_num == tp[i][0] // 24
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=VARS[0]['H'].sel(time=tp[i][0]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
                
                else:
                    if day_num == tp[i] // 24:
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=VARS[0]['H'].sel(time=tp[i]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
            
        else:
            day_num = 0
             # starting hydrogen levels
            
            Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
            #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
            Hs[0,:]=VARS[0]['H'].sel(time=0).max('scenario') # else might get unfeasibility for net 0 solutions
        # start iterating
        for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
            if day_num < 365:    
                #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
                EW = wind_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
                ES = pv_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                HL = hydrogen_demand_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                EL = elec_load_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                
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
                c1 = model.addConstrs((- H[i,k] + H[i-1,k] + 30*network.n['feth'].iloc[k]*EtH[i,k] - HtE[i,k] -
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = i, node = k) for i in range(1,inst) for k in range(Nnodes)))
            
            
                c2 = model.addConstrs((ns[k]*float(ES.isel(time = i, node = k)) + nw[k]*float(EW.isel(time = i, node = k)) + 0.033*network.n['fhte'].iloc[k]*HtE[i,k] - EtH[i,k]
                                    - quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                                    + quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    >= EL.isel(time = i, node = k) for k in range(Nnodes)for i in range(inst)))
                
                # constrain to end of previous day
                #print([Hs.isel(time = -1,node = k) for k in range(Nnodes)])
                c3 = model.addConstrs((- H[0,k] + Hs.isel(time = -1,node = k) + 30*network.n['feth'].iloc[k]*EtH[0,k] - HtE[0,k] -
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = 0, node = k) for k in range(Nnodes)))

                
                # constraints for loss:
                #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                #c4 = model.addConstrs( delta_H[i,k] >= - H[i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for i in range(inst) for k in range(Nnodes))
                
                
                
                model.optimize()
                if model.Status!=2:
                    print("Status = {}".format(model.Status))
                    print(f"Failed at day {day_num}, scenario {j}")
                    return day_num, j
                else:
                    values = np.zeros([inst,len(network.n.index.to_list())])
                    for key in H:
                        #print(key)
                        values[key]=H[key].X
                    
                    Hss = xr.DataArray(values, 
                                        dims=["time","node"], 
                                        coords = dict(zip(["time","node"],[range(inst),network.n.index.to_list()])))
                    Hs = xr.concat([Hs,Hss],dim='time')
                    print('validation opt time: ',np.round(time.time()-start_time,4),'s. Day: ',day_num, "Scenario: ", j)

            else: #if we are in the last day
                break
    return day_num, j

#%%
def Validate_mhte(network,VARS,scenarios,free_mhte=True):
         
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
    d = scenarios['wind_scenario'].shape[2]
    inst = 24
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,ub=mhte) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, ub=meth) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,ub=nh)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    # model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    # model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
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
    
   
    day_num = 0
    MHTE=mhte
    METH=meth
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
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
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            break
        else:
            
            max_mhte=max([max([max([HtE[j,i,k].X for i in range(inst)]) for j in range(d)]) for k in range(Nnodes)])
            max_meth=max([max([max([EtH[j,i,k].X for i in range(inst)]) for j in range(d)]) for k in range(Nnodes)])
            if MHTE<max_mhte:
                MHTE=max_mhte
            if METH<max_meth:
                METH=max_meth
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return {'day_num':day_num, 'Hs':Hs, 'MHTE':MHTE,'METH':METH}


#%% import_generated_scenario

def import_scenario_val(start,stop):
    
    path = "model/scenario_generation/scenarios/"
    elec_load_df = pd.read_csv(path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR','DE']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR','DE']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France','Germany'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    
    ave = [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229 ]
    a = xr.DataArray(ave,dims=['node'], coords={'node':['Italy','Spain','Austria','France','Germany']})
    elec_load_scenario=elec_load_scenario*a
    
    #wind_scenario = 4*scenario_to_array(pd.read_csv(path +'small-eu-wind-scenarios3.csv', index_col = 0))
    wind_scenario = 4*import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = 0.01*import_scenario(path + 'small-eu-PV-scenarios.csv')
    
    
    df = pd.read_csv(path+'hydrogen_demandg.csv', index_col = 0).head()
    time_index = range(df.shape[1])#pd.date_range('2023-01-01 00:00:00', periods=df.shape[1], freq='H')
    node_names=['Italy', 'Spain', 'Austria', 'France','Germany']

    hydro = xr.DataArray(
        np.expand_dims(df.T.values, axis = 2),
        coords={'time': time_index, 'node': node_names, 'scenario': [0]},
        dims=['time', 'node', 'scenario'] )
    hydro_mean = hydro.mean(dim = ["time","scenario"]) 
    hydrogen_demand_scenario = hydro / hydro_mean
    
    
    #hydrogen_demand_scenario2 = import_scenario(path + 'hydrogen_demandg.csv')
    
    #eu.add_scenarios(wind_scenario * max_wind, pv_scenario * max_solar, hydrogen_demand_scenario, elec_load_scenario)
    
    scenarios = {
        'wind_scenario' : wind_scenario.sel(scenario=slice(start,stop)),
        'pv_scenario' : pv_scenario.sel(scenario=slice(start,stop)),
        'hydrogen_demand_scenario' : hydrogen_demand_scenario,
        'elec_load_scenario' : elec_load_scenario
        }
    
    
    return scenarios

def df_aggregator2(network, df0, prev_df, splitted_intervals, son_indeces_lists):
    """
    Aggregates the data in the given DataFrame `df` based on the time aggregation specified in `network.time_partition`.
    
    Parameters:
        network (Network): The network object containing the time partition specification.
        df0 (xarray.DataArray): The original DataFrame containing the data to be aggregated.
        prev_df (xarray.DataArray): The previous aggregation of df0, respect to the partition old_tp.
        splitted_intervals (list): A list of time intervals that were split from the original time partition.
        son_indeces_lists (list): A list of lists containing the indices of the son intervals in the time partition.
    
    Returns:
        xarray.DataArray: The aggregated DataFrame.
    """
    df00 = df0.assign_coords({'time': range(df0.shape[0])})
    tp_obj = network.time_partition
    old_tp = tp_obj.old_agg[-1]
    tp = tp_obj.agg
    father_indices = [old_tp.index(i) for i in splitted_intervals if i in old_tp]

    slices = []
    
    # Add initial slice
    if father_indices[0] > 0:
        slices.append(prev_df.sel(time=slice(None, father_indices[0]-1)).drop_vars('time', errors='ignore'))
    
    for i in range(len(father_indices) - 1):
        son_indices = son_indeces_lists[i]
        
        # Add son indices slices
        for t in (tp[j] for j in son_indices):
            if isinstance(t, list):
                slices.append(df00.sel(time=t).sum(dim='time').drop_vars('time', errors='ignore'))
            else:
                slices.append(df00.sel(time=t).drop_vars('time', errors='ignore'))
        
        # Add slice between current and next father index
        if father_indices[i] + 1 <= father_indices[i + 1] - 1:
            slices.append(prev_df.sel(time=slice(father_indices[i] + 1, father_indices[i + 1] - 1)).drop_vars('time', errors='ignore'))
    
    # Add final son indices slices
    son_indices = son_indeces_lists[-1]
    for t in (tp[i] for i in son_indices):
        if isinstance(t, list):
            slices.append(df00.sel(time=t).sum(dim='time').drop_vars('time', errors='ignore'))
        else:
            slices.append(df00.sel(time=t).drop_vars('time', errors='ignore'))
    
    # Add final slice
    if father_indices[-1] + 1 < len(prev_df['time']):
        slices.append(prev_df.sel(time=slice(father_indices[-1] + 1, None)).drop_vars('time', errors='ignore'))

    # Concatenate all slices
    new_df = xr.concat(slices, dim='time', coords='minimal', compat='override')

    # Assign new time coordinates
    new_time_coords = range(len(new_df['time']))
    new_df = new_df.assign_coords(time=('time', new_time_coords))
    
    return new_df


def get_rho(n,VARS, n_max = 10):
    """
    given a network and a solution to an aggregated problem, calculates the rho value for each constraint
    """
    #rho for a constraint
    tp = n.time_partition.agg
    HL = n.loadH_t
    PL = n.loadP_t
    ES = n.genS_t
    EW = n.genW_t
    nw = xr.DataArray(VARS["nw"], dims='node', coords={'node': HL.coords['node']})
    ns = xr.DataArray(VARS["ns"], dims='node', coords={'node': HL.coords['node']})
    ES = ES.assign_coords(time=PL.coords['time'])
    EW = EW.assign_coords(time=PL.coords['time'])
    Pnett = PL - nw*EW - ns*ES 
    Hnett = HL #we don't haev any hydrogen generations corresponding to time independent variables
    pt = n.time_partition.partitionize(tp)
    intervals = [i for L in [[k]*len(pt[k]) for k in range(len(tp))] for i in L]
    Pnett = Pnett.assign_coords(interval=('time',intervals))
    Hnett = Hnett.assign_coords(interval=('time',intervals))
        #P_net.append(Pnett)
        #H_net.append(Hnett)

    rhoP = Pnett.groupby('interval') / (Pnett.groupby('interval').sum())
    rhoH = Hnett.groupby('interval') / (Hnett.groupby('interval').sum())
    varhop = rhoP.var(dim='node')
    varhoh = rhoH.var(dim='node')
    varho = varhop + varhoh

    
    return rhoP, rhoH, varho
# 'top_n_coords' now contains the coordinates of the top 'n' values

def xr_top_n(x, n_max, dim = 'time'):
    # Stack all dimensions into a single one for easier sorting
    stacked = x.stack(all_dims=x.dims).copy()
    # Find the coordinates of the top 'n' values
    top_n_coords = {dim: []}
    max_values = []
    for _ in range(n_max):
        # Get the position of the maximum value
        max_idx = stacked.argmax('all_dims')
        max_value = stacked.isel(all_dims=max_idx).values.item()
        max_values.append(max_value)
        # Get the coordinates corresponding to the maximum value
        max_coords = stacked.isel(all_dims=max_idx).coords
        # Store the coordinates
        top_n_coords[dim].append(int(max_coords[dim]))
        # Set the current maximum value to a very low value to exclude it in the next iteration
        stacked[max_idx] = float('-inf')

    return top_n_coords

# %%
#LESS IMPORTANT:
# give name to constriants
#todo: da spostrare in uno script più sensato
#todo: import classic example cases
#todo: some time aggregation plotting

def solution_to_xarray(var, dims, coords):
    values = np.zeros(tuple([len(c) for c in coords]))
    for key in var.keys():
        values[key] = var[key].X
    
    array = xr.DataArray(values,
                        dims=dims,
                        coords = dict(zip(dims,coords)))
    return array



#%% class time_partition

class time_partition:

    def aggregate(self,l, i0,i1):
        """
        Aggregates a list into sublists based on the given indices.

        Args:
            l (list): The list to be aggregated.
            i0 (int): The starting index of the sublist.
            i1 (int): The ending index of the sublist.

        Returns:
            list: The aggregated list with sublists.
        """
        if i0 == 0:
            return [l[i0:i1]]+ l[i1:]
        elif i1 == len(l):
            return l[0:i0]+[l[i0:i1]]
        elif i0 == i1:
            return l
        else:
            return l[0:i0]+[l[i0:i1]]+ l[i1:]
    def disaggregate(self,l,i):
        if type(l[i]) is list:
            return l[0:i] + l[i] + l[i+1:]
        else:
            raise ValueError(f"{l[i]} is not a list and cannot be disaggregated")

    def day_aggregation(self):
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        for day in range(n_days): #ragruppo i giorni
            l = self.aggregate(l, day, day + 24)

        n_seasons = 16
        season = int(np.floor(n_days / n_seasons))  #mettiamo 10 giorni interi per intero
        for i in range(n_seasons):
            l = self.disaggregate(l, i * season + 23*i)
        return l

    def day_night_aggregation(self):
        """
        Aggregates the time steps into sublists based on day and night periods.

        Returns:
            list: The aggregated list with sublists representing day and night periods.
        """
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        l = self.aggregate(l, 0, 6) #initial night
        for day in range(n_days-1): #ragruppo i giorni
            l = self.aggregate(l, 2*day + 1, 2*day + 1 + 14)
            l = self.aggregate(l, 2*day + 2, 2*day + 2 + 10) 
        day = n_days-1
        l = self.aggregate(l, 2*day + 1, 3*day + 1 + 14)
        l = self.aggregate(l, 2*day + 2, 3*day + 2 + 4)
        l1 = []
        for i in l:
            if type(i) is int: #ho modificato qui protrebbe creare problemi, stavo cercando di togliere liste vuote.
                l1 += i
            elif len(i) > 1:
                l1.append(i)
        l = l1
        #season = int(np.floor(n_days /10))  #mettiamo 10 giorni interi per intero
        #for i in range(10):
        #    l = self.disaggregate(l, i * season + 23*i)
        return l

    def __init__(self,T, init_method='day_aggregation'):
        self.T=T
        self.time_steps = list(range(T))
        #print(init_method)
        #define initial_aggregation with the chosen method.
        method = getattr(self, init_method)
        if method is not None and callable(method):
            self.agg = method()
        else:
            print(f"Method {init_method} not found or not callable")
        
        self.old_agg = []
        self.family_tree = [] #list of lists of intervals splitted in some way (todo? become a dictionary mapping which interval in splitted into which intervals)

    def len(self,i):
        if type(self.agg[i]) is list:
            return len(self.agg[i])
        else:
            return 1

    def random_iter_partition(self,k=1):
        self.old_agg += [self.agg.copy()]
        family_list = []
        for _ in range(k):
            tp = self.agg
            agg_indices = [i for i in range(len(tp)) if type(tp[i]) is list] #get index of aggregate intervals
            rand_ind = agg_indices[np.random.randint(len(agg_indices))] #get random index
            new_int =self.agg[rand_ind]
            self.agg = self.disaggregate(self.agg,rand_ind)
            family_list += [new_int]
        self.family_tree += [time_partition.order_intervals(family_list)]

    def iter_partition(self, t):
        self.old_agg += [self.agg.copy()]
        family_list = []
        tp = self.agg
        new_int =self.agg[t]
        if type(new_int) is list:
            self.agg = self.disaggregate(self.agg,t)
            family_list += [new_int]
            self.family_tree += [time_partition.order_intervals(family_list)]
        else:
            print(f"{new_int} is a singleton")
    
    def iter_partition_intervals(tp_obj, intervals):
        "completly disaggregates intervals in intervals"
        tp_obj.old_agg += [tp_obj.agg.copy()]
        family_list = []
        intervals = sorted(intervals)[::-1] #reverse intervals order so that disaggregating does not change the indexes.
        for t in intervals:
            new_int =tp_obj.agg[t]
            if type(new_int) is list:
                tp_obj.agg = tp_obj.disaggregate(tp_obj.agg,t)
                family_list += [new_int]
            else:
                print(f"{new_int} is a singleton")
        if len(family_list) > 0:
            #print("appending new intervals to family tree")
            tp_obj.family_tree += [tp_obj.order_intervals(family_list)]
        else:
            print("No intervals where splitted, iteration left partion identical")

    def to_dict(self):
        """
        Save the Network object as a dictionary of dictionaries, where each attribute is stored as a dictionary.

        Returns:
            dict: A dictionary containing all the attributes of the Network object.
        """
        output = {}
        for attr in self.__dict__:
            attribute = getattr(self, attr)
            if type(attribute) is int:
                output[attr] = attribute
            elif type(attribute) is float:
                output[attr] = attribute
            elif type(attribute) is str:
                output[attr] = attribute
            elif type(attribute) is list:
                output[attr] = attribute
            else:
                output[attr] = getattr(self, attr).to_dict()
        return output

    @staticmethod
    def from_dict(d):
        tp = time_partition(d['T'])
        for attr, value in d.items():
            setattr(tp, attr, value)
        return tp

    @staticmethod
    def tuplize(tp):
        l = []
        for i in tp:
            if type(i) is list:
                l.append(tuple(i))
            else:
                l.append(i)
        return l

    @staticmethod
    def partitionize(tp):
        l = []
        for i in tp:
            if type(i) is int:
                l.append([i])
            else:
                l.append(i)
        return l

    @staticmethod
    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

    @staticmethod
    def interval_subsets(interval,tp):
        """
        Get the indexes of elements in the given time partition `tp` that are subsets of the given `interval`.

        Parameters:
            interval (set): The set of elements to check for subset membership.
            tp (list): The list of elements to search for subsets of `interval`.

        Returns:
            list: The indexes of elements in `tp` that are subsets of `interval`.
        """
        indexes = []
        for i in range(len(tp)):
            l = tp[i]
            if type(l) is list:
                if set(l).issubset(interval):
                    indexes.append(i)
            else:
                if l in interval:
                    indexes.append(i)
        return indexes
        
# %% df_aggregator

def df_aggregator(df, time_partition):
    """
    Aggregates the data in the given DataFrame `df` based on the time aggregation specified in `agg_time`.
    In particular it sums the coordinatees found in the same time interbals in time_partition together
    Parameters:
        df (xarray.DataArray): The DataFrame containing the data to be aggregated.
        time_partition (list): A list of time aggregation specifications. Each element in the list can be either a single time value or a list of time values.
            
    """
    df2 = df.assign_coords({'time': range(df.shape[0])})
    summed_df = []
    for t in time_partition:
        if type(t) is list:
            summed_df.append(df2.sel(time = t).sum(dim='time'))
        else:
            summed_df.append(df2.sel(time = t).drop_vars('time', errors='ignore'))

    add_df = xr.concat(summed_df, dim = 'time', coords = 'minimal', compat = 'override').assign_coords(time = ('time', range(len(time_partition))))

    return add_df



#%% class Network

class Network:
    """
    Class to represent a network of nodes and edges.
    
    Attributes:
        n (pandas.DataFrame): DataFrame with index name of node, columns lat and long.
        edgesP (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        edgesH (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        loadH (pandas.DataFrame): DataFrame with time dependent variables for hydrogen.
        loadE (pandas.DataFrame): DataFrame with time dependent variables for electricity.
        genW (pandas.DataFrame): DataFrame with time dependent variables for wind.
        genS (pandas.DataFrame): DataFrame with time dependent variables for solar.
    """
    
    def __init__(self, init_method = 'day_night_aggregation' ):
        """
        Initialize a Network object.
        
        Parameters:
            nodes (pandas.DataFrame): DataFrame with index name of node, columns lat and long.
            edgesP (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
            edgesH (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        """
        self.init_method = init_method
        self.n=pd.DataFrame(columns=['node','location','lat','long', 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh','MP_wind','MP_solar','meanP_load','meanH_load']).set_index('node')
        # nodes = pd.DataFrame with index name of node, columns lat and long.
        self.edgesP=pd.DataFrame(columns=['start_node','end_node','NTC'])
        self.edgesH=pd.DataFrame(columns=['start_node','end_node','MH'])
        # edges = pd.DataFrame with columns start node and end node, start coords and end coords.
        self.n_scenarios = 0
        self.costs = pd.DataFrame(columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])

    def add_node(self, new_df ):
        if 'node' in new_df.columns:
            self.n.loc = pd.concat(self.n, new_df.set_index('node'))
        else:
             self.n.loc = pd.condat(self.n, new_df.set_index('node'))

    def add_scenarios(self, wind_scenario, pv_scenario, hydrogen_demand_scenario, elec_load_scenario):
        n_scenarios = self.n_scenarios
        self.genW_t = wind_scenario.sel(scenario=slice(0,n_scenarios-1))
        self.genS_t = pv_scenario.sel(scenario=slice(0,n_scenarios-1))
        self.loadH_t = hydrogen_demand_scenario.sel(scenario=slice(0,n_scenarios-1)) 
        self.loadP_t = elec_load_scenario.sel(scenario=slice(0,n_scenarios-1)) 

        self.T = self.genW_t.shape[0]
        self.d = self.genW_t.shape[2]
        
    def init_time_partition(self):
        """
        Initializes the time partition for the Network.
        This method creates a `time_partition` object using the `time_partition` class and assigns it to the `self.time_partition` attribute. It then retrieves the aggregation specification from the `agg` attribute of the `time_partition` object.
        The method then uses the `df_aggregator` function to aggregate the `genW_t`, `genS_t`, `loadH_t`, and `loadP_t` attributes of the `self` object based on the aggregation specification. The aggregated data is assigned to the corresponding attributes of the `self` object.
        """
        self.time_partition = time_partition(self.T, self.init_method)
        agg = self.time_partition.agg
        self.genW_t_agg = df_aggregator(self.genW_t, agg)
        self.genS_t_agg = df_aggregator(self.genS_t, agg)
        self.loadH_t_agg = df_aggregator(self.loadH_t, agg)
        self.loadP_t_agg = df_aggregator(self.loadP_t, agg)
   
    #network, df0, df, splitted_intervals, son_indeces_lists
    def update_time_partition(self):
        #print("updating time partition")
        tp_obj = self.time_partition
        tp = tp_obj.agg
        splitted_intervals = tp_obj.family_tree[-1]
        son_indeces_lists = [tp_obj.interval_subsets(father_interval, tp) for father_interval in splitted_intervals]

        df0 = self.genW_t
        df = self.genW_t_agg
       
        self.genW_t_agg = df_aggregator2(self, self.genW_t, self.genW_t_agg, splitted_intervals, son_indeces_lists)
        self.genS_t_agg =df_aggregator2(self,self.genS_t, self.genS_t_agg, splitted_intervals, son_indeces_lists)
        self.loadH_t_agg = df_aggregator2(self,self.loadH_t, self.loadH_t_agg, splitted_intervals, son_indeces_lists)
        self.loadP_t_agg = df_aggregator2(self,self.loadP_t, self.loadP_t_agg, splitted_intervals, son_indeces_lists)
        #print(f"new dataframe shape: {self.genW_t_agg.shape}, len tp {len(tp)}")

    def iter_partition(self,k=1):
        """
        Iteratively partitions the time intervals in the network.

        Parameters:
            k (int): Number of iterations.

        Returns:
            None
        """
        #print("iterating")
        self.time_partition.random_iter_partition(k)
        self.update_time_partition()

    def rho_iter_partition(self,VARS, k=1):
        #"a posteriori iteration method"
        #print("iterating")
        rhoP, rhoH, varho = get_rho(self, VARS)
        tp = self.time_partition.agg
        varho_grpd =varho.groupby('interval').sum()#drop singletons intervals
        varho_grpd = varho_grpd.where(varho_grpd['interval'].isin([k for k in range(len(tp)) if type(tp[k]) is list]), drop = True) 
        top_n_intervals = xr_top_n(varho_grpd, k, dim='interval')['interval']
        ##print(top_n_intervals)
        self.time_partition.iter_partition_intervals(top_n_intervals)
       #TODO: uncomment 
        self.update_time_partition()

    def fail_int(self, fail):
        """
        This function determines in which time interval the validation failed based on the input fail value.

        Parameters:
            fail (int): The day in which the validation failed.

        Returns:
            None
        """
        tp = self.time_partition.agg
        fail_int = 0
        fail_hour = fail*24
        #find in which time interval the validation failed
        for t in tp:
            if type(t) is list:
                if fail_hour <= t[0] and t[0] < fail_hour + 24: #look for an interval which is in the fail day
                    break
            else:
                if fail_hour + 24 <= t:
                    break
            fail_int += 1

        return fail_int

    def validationfun_iter_partition(self, VARS, k=1):
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t

        fail = Validate(self,[VARS],scenarios) #initial hour of the validation on which it fails
        if fail < 365:
           
            fail_int = self.fail_int(fail)
            logging.debug(f'debug: fail_interval is {fail_int}, {tp[fail_int]} on day {fail}')
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)

            self.update_time_partition()
            return False
        else:
            return True
    def validationfun_iter_partition_old(self, VARS, k=1):
        """
        Validates the current time partition and updates it based on the validation result.

        Parameters:
            VARS: List of variables required for validation.
            k (int, optional): Number of iterations for random partitioning. Default is 1.

        Returns:
            bool: True if validation is successful, False otherwise. If validation fails,
                the function attempts to find the time interval in which it failed and
                disaggregate it. If no interval is found, it performs random iteration.
        """
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t
        fail = Validate(self,[VARS],scenarios)
        if fail < 365:
            fail_int = 0
            #find in which time interval the validation failed
            for t in tp:
                if type(t) is list:
                    if fail in t:
                        break
                else:
                    if fail == t:
                        break
                fail_int += 1
            
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)
            self.update_time_partition()
            return False
        else:
            return True
    def validation2fun_iter_partition(self, VARS, k=1, day_initial = 0, scenario_initial = 0):
        tp = self.time_partition.agg
        
        scenarios = {}
        scenarios['wind_scenario'] = self.genW_t
        scenarios['pv_scenario'] = self.genS_t
        scenarios['hydrogen_demand_scenario'] = self.loadH_t
        scenarios['elec_load_scenario'] = self.loadP_t

        fail_day, fail_scen = Validate2(self,[VARS],scenarios, day_initial, scenario_initial) #initial hour of the validation on which it fails
        if fail_day < 364 or fail_scen < self.d - 1:
           
            fail_int = self.fail_int(fail_day)
            logging.debug(f'debug: fail_interval is {fail_int}, {tp[fail_int]} on day {fail_day}')
            #if it's an interval, disgregate it.
            if type(tp[fail_int]) is list:
                print("disaggregating interval")
                self.time_partition.iter_partition(fail_int)
            else:
                print("no interval to disaggregate, iterating randomly")
                self.time_partition.random_iter_partition(k)

            self.update_time_partition()
            return False, fail_day, fail_scen
        else:
            return True, fail_day, fail_scen

      



    def plot(self):
        """
        Plot the network using folium.
        """
        if 'node' in self.n.columns:
            self.n.set_index('node', inplace=True)
        # Filter out rows with NaN values in 'lat' or 'long'
        valid_nodes = self.n.dropna(subset=['lat', 'long'])
        #print("valid nodes",valid_nodes.index)
        loc = [valid_nodes['lat'].mean(), valid_nodes["long"].mean()]
        m = folium.Map(location=loc, zoom_start=5, tiles='CartoDB positron')
        
        for node in valid_nodes.index.to_list():
            folium.Marker(
                location=(valid_nodes.loc[node, 'lat'], valid_nodes.loc[node, 'long']),
                icon=folium.Icon(color="green"),
            ).add_to(m)
        
        for edge in self.edgesP.index.to_list():
            start_node, end_node = self.edgesP.loc[edge, 'start_node'], self.edgesP.loc[edge, 'end_node']
            if start_node in valid_nodes.index and end_node in valid_nodes.index:
                start_loc = (valid_nodes.loc[start_node, 'lat'], valid_nodes.loc[start_node, 'long'])
                end_loc = (valid_nodes.loc[end_node, 'lat'], valid_nodes.loc[end_node, 'long'])
                folium.PolyLine([start_loc, end_loc], weight=5, color='blue', opacity=.2).add_to(m)
            else:
                print("node not valid edge",start_node,end_node)
                print(self.n)
        
        self.n.reset_index(inplace=True)
        return m._repr_html_()

    def get_edgesP(self,start_node,end_node):
        return self.edgesP[self.edgesP['start_node']==start_node][self.edgesP['end_node']==end_node]

    def get_edgesH(self,start_node,end_node):
        return self.edgesH[self.edgesH['start_node']==start_node][self.edgesH['end_node']==end_node]
    

    def neighbour_edgesP(self,direction, node_index):
        """
        Returns a list of edge indices that are connected to a given node in a specified direction.

        Parameters:
            direction (str): The direction of the edges to be retrieved. '+' for outgoing edges and '-' for incoming edges.
            node_index (int): The index of the node in the graph.

        Returns:
            list: A list of edge indices that are connected to the given node in the specified direction.
        """
        if direction == '+':
            return self.edgesP.loc[self.edgesP['start_node']==self.n.index.to_list()[node_index]].index.to_list()
        if direction == '-':
            return self.edgesP.loc[self.edgesP['end_node']==self.n.index.to_list()[node_index]].index.to_list()

    def neighbour_edgesH(self,direction, node_index):
        if direction == '+':
            return self.edgesH.loc[self.edgesH['start_node']==self.n.index.to_list()[node_index]].index.to_list()
        if direction == '-':
            return self.edgesH.loc[self.edgesH['end_node']==self.n.index.to_list()[node_index]].index.to_list()

    def to_dict(self):
        """
        Save the Network object as a dictionary of dictionaries, where each attribute is stored as a dictionary.

        Returns:
            dict: A dictionary containing all the attributes of the Network object.
        """
        output = {}
        if 'node' in self.n.columns:
            self.n.set_index('node', inplace=True)
        for attr in self.__dict__:
            attribute = getattr(self, attr)
            if type(attribute) is int:
                output[attr] = attribute
            elif type(attribute) is float:
                output[attr] = attribute
            elif type(attribute) is str:
                output[attr] = attribute
            elif type(attribute) is dict:
                output[attr] = attribute
            elif type(attribute) is list:
                output[attr] = attribute
            elif attr == 'time_partition':
                output[attr] = attribute.to_dict()
            else:
                output[attr] = getattr(self, attr).to_dict()
        return output
    
    @staticmethod
    def from_dict(dictionary):
        '''
        converts a dictionary to a Network object
        '''
        n = Network()
        for attr, value in dictionary.items():
            if type(value) is int:
                setattr(n, attr, value)
            elif type(value) is float:
                setattr(n, attr, value)
            elif type(value) is str:
                setattr(n, attr, value)
            elif type(value) is list:
                setattr(n, attr, value)
            elif attr == 'time_partition':
                tp = time_partition.from_dict(value)
                setattr(n, attr, tp)
            elif type(value) is dict:
                if len(value) == 0:
                    setattr(n, attr, pd.DataFrame(value))
                elif attr in ["n","edgesP","edgesH","costs"]:
                    ##print(f"it should become a dataframe but let's see, {attr}")
                    max_len = max([len(val) for key, val in value.items()])
                    for key, val in value.items():
                        if len(val) < max_len:
                            value[key] = np.concatenate((val, np.full((max_len - len(val)),np.nan)))
                    setattr(n, attr, pd.DataFrame(value))     
                else:
                    ##print(f"it should become a dataarray but let's see, {attr}")
                    da = xr.DataArray.from_dict(value)
                    setattr(n, attr, da)
            else:
                print(f"Some attribute is not supported: {attr} has value of type: {type(value)}")
        
        #n.n = n.n.set_index('node')
        if type(n.edgesP.index[0]) == str:
            n.edgesP.index = n.edgesP.index.astype(int)
            n.edgesH.index = n.edgesH.index.astype(int)

        return n



# %%
