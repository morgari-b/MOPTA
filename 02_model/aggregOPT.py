# %%
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
from YUPPY import OPT1, OPT2, Network
from EU_net import EU
#os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")

# %% class time aggregator
class time_aggregator:
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
            return l  

    def initial_aggregation(self):
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        for day in range(n_days): #ragruppo i giorni
            l = self.aggregate(l, day, day + 24)
        
       
        season = int(np.floor(n_days /10))  #mettiamo 10 giorni interi per intero
        for i in range(10):
            l = self.disaggregate(l, i * season + 23*i)
        return l

    def __init__(self,T):
        self.T=T
        self.time_steps = list(range(T))
        self.agg = self.initial_aggregation() #define initial aggregation

def df_aggregator(df, time_partition):
    """
    Aggregates the data in the given DataFrame `df` based on the time aggregation specified in `agg_time`.
    In particular it sums the coordinatees found in the same time interbals in time_partition together
    Parameters:
        df (xarray.DataArray): The DataFrame containing the data to be aggregated.
        time_partition (list): A list of time aggregation specifications. Each element in the list can be either a single time value or a list of time values.
            
    """
    summed_df = []
    for t in time_partition:
        if type(t) is list:
            summed_df.append(df.sel(time = t).sum(dim='time'))
        else:
            summed_df.append(df.sel(time = t).drop_vars('time', errors='ignore'))

    add_df = xr.concat(summed_df, dim = 'time', coords = 'minimal', compat = 'override').assign_coords(time = ('time', range(len(time_partition))))

    return add_df



def OPT3(Network):
    if Network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    

    start_time=time.time()
    Nnodes = Network.n.shape[0]
    NEedges = Network.edgesP.shape[0]
    NHedges = Network.edgesH.shape[0]
    d = Network.loadP_t.shape[2] #number of scenarios
    inst = Network.loadP_t.shape[0] #number of time steps T
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    
    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=Network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=Network.n['Mnw'])    
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=Network.n['Mnh'])   
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=Network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=Network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC) 
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH) 
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with Network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes)) 
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= Network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= Network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))

    outputs=[]
    VARS=[]
    #todo perchè lo rimuoviamo questo?
    cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
    #todo: sobstitute 30 with a parameter
    cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*Network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -  
                           quicksum(H_edge[j,i,l] for l in Network.edgesH.loc[Network.edgesH['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,i,l] for l in Network.edgesH.loc[Network.edgesH['end_node']==Network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*Network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
                           quicksum(H_edge[j,inst-1,l] for l in Network.edgesH.loc[Network.edgesH['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,inst-1,l] for l in Network.edgesH.loc[Network.edgesH['end_node']==Network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for k in range(Nnodes))
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    
    
    ES = Network.genS_t
    EW = Network.genW_t
    EL = Network.loadP_t
    HL = Network.loadH_t

    model.remove(cons1)
    for j in range(d): 
        for k in range(Nnodes):
            for i in range(inst-1):
                cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel     
            cons3[j,k].rhs  = HL[inst-1,k,j]
    
    try:    
        cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*Network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['end_node']==Network.n.index.to_list()[k]].index.to_list()) 
                            >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst))
    except IndexError as e:
        print(f"IndexError occurred at i={i}, j={j}, k={k}")
        print(f"ES shape: {ES.shape}")
        print(f"EW shape: {EW.shape}")
        print(f"HtE shape: {HtE.shape}")
        print(f"EtH shape: {EtH.shape}")
        print(f"P_edge shape: {P_edge.shape}")
        print(f"Network.n indices: {Network.n.index.to_list()}")
        print(f"Network.edgesP start_node indices: {Network.edgesP['start_node'].index.to_list()}")
        print(f"Network.edgesP end_node indices: {Network.edgesP['end_node'].index.to_list()}")
        raise e  # Re-raise the exception after logging the details
    
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        VARS=[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]       
        outputs=outputs + [VARS+[model.ObjVal]] 
        print("opt time: {}s.".format(np.round(time.time()-start_time,3)))
            
    return outputs#,HH,ETH,HTE
# %%
eu = EU()
notagg_results = OPT3(eu)
T =  eu.genW_t.shape[0]
agg = time_aggregator(T)
time_partition = agg.agg

#summed_df = [df.sel(time = t).sum(dim='time') for t in agg_time]

df_aggregator(df, time_partition)
# %%

eu.genS_t = df_aggregator(eu.genS_t, time_partition)
eu.genW_t = df_aggregator(eu.genW_t, time_partition)
eu.loadP_t = df_aggregator(eu.loadP_t, time_partition)
eu.loadH_t = df_aggregator(eu.loadH_t, time_partition)

agg_results = OPT3(eu)
# %%
