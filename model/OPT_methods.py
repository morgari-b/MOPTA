
# %% import
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
from model.YUPPY import Network, time_partition, df_aggregator, solution_to_xarray
from model.EU_net import EU
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px



#%% OPT1 - single node
def OPT1(es,ew,el,hl,d=5,rounds=4,cs=4000, cw=3000000,ch=10,Mns=10**5,Mnw=500,Mnh=10**9,chte=2,fhte=0.75,Mhte=10**6,ceth=200,feth=0.7,Meth=10**5):
            
    start_time=time.time()
    
    D,inst = np.shape(es)
    rounds=min(rounds,D//d)
    #print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    max_wind = 4 * 1000
    max_solar = 10
    ns = model.addVar(vtype=GRB.CONTINUOUS, obj=cs,ub=Mns)
    nw = model.addVar(vtype=GRB.CONTINUOUS, obj=cw,ub=Mnw)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,ub=Mnh)   
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01, ub=Mhte)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01,ub=Meth)
    
    HtE = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)

    model.addConstrs( H[j,i] <= nh for i in range(inst) for j in range(d))
    model.addConstrs( EtH[j,i] <= meth for i in range(inst) for j in range(d))
    model.addConstrs( HtE[j,i] <= mhte for i in range(inst) for j in range(d))

    outputs=[]
    VARS=[]
    cons1=model.addConstrs(nh>=0 for j in range(d) for i in range(inst))
    cons2=model.addConstrs(- H[j,i+1] + H[j,i] + 30*feth*EtH[j,i] - HtE[j,i]==0 for j in range(d) for i in range(inst-1))
    cons3=model.addConstrs(- H[j,0] + H[j,inst-1] + 30*feth*EtH[j,inst-1] - HtE[j,inst-1] == 0 for j in range(d))
    
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        gr_start_time=time.time()

        ES=es[d*group:d*group+d,:]
        EW=ew[d*group:d*group+d,:]
        EL=el[d*group:d*group+d,:]
        HL=hl[d*group:d*group+d,:]

        model.remove(cons1)
        for j in range(d): 
            for i in range(inst-1):
                cons2[j,i].rhs = HL[j,i]
            cons3[j].rhs  = HL[j,inst-1]
        cons1=model.addConstrs(ns*ES[j,i] + nw*EW[j,i] + 0.033*fhte*HtE[j,i] - EtH[j,i] >= EL[j,i] for j in range(d) for i in range(inst))
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[np.ceil(ns.X),np.ceil(nw.X),nh.X,mhte.X,meth.X]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            #print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE


    
#%% OPT2 - network
def OPT2(network, d=1,rounds=1,long_outs=False):
    
    if network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, ch_t, chte, ceth, cNTC, cMH = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    

    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    D = network.loadP_t.shape[2] #number of scenarios
    inst = network.loadP_t.shape[0] #number of time steps T
    rounds=min(rounds,D//d)
    #print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    
    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])    
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])   
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC) 
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH) 
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ch_t/d, lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes)) 
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))

    outputs=[]
    VARS=[]
    HX=network.genW_t.copy()
    HX[:,:,:]=0
    EtHX=HX.copy()
    HtEX=HX.copy()
    P_edgeX=HX.copy()
    H_edgeX=HX.copy()
    
    #todo perchè lo rimuoviamo questo? # perché è quello che va sostituito in toto ad ogni giro... è qui come fermaposto
    cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
    #todo: sobstitute 30 with a parameter # NO!! 30 e poi sotto 0.033 sono i rate di conversione kg di idrogeno - MW se la conversione fosse 100% efficiente, non sono parametri.
    cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -  
                           quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
                           quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for k in range(Nnodes))
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        gr_start_time=time.time()

        ES = network.genS_t.sel(scenario = slice(d*group, d*(group+1)))
        EW = network.genW_t.sel(scenario = slice(d*group, d*(group+1)))
        EL = network.loadP_t.sel(scenario = slice(d*group, d*(group+1)))
        HL = network.loadH_t.sel(scenario = slice(d*group, d*(group+1)))

        model.remove(cons1)
        for j in range(d): 
            for k in range(Nnodes):
                for i in range(inst-1):
                    cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel     
                cons3[j,k].rhs  = HL[inst-1,k,j]
        
        try:    
            cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list()) 
                                >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst))
        except IndexError as e:
            print(f"IndexError occurred at i={i}, j={j}, k={k}")
            print(f"ES shape: {ES.shape}")
            print(f"EW shape: {EW.shape}")
            print(f"HtE shape: {HtE.shape}")
            print(f"EtH shape: {EtH.shape}")
            print(f"P_edge shape: {P_edge.shape}")
            print(f"network.n indices: {network.n.index.to_list()}")
            print(f"network.edgesP start_node indices: {network.edgesP['start_node'].index.to_list()}")
            print(f"network.edgesP end_node indices: {network.edgesP['end_node'].index.to_list()}")
            raise e  # Re-raise the exception after logging the details
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
        if long_outs==True:
            for i in range(inst):
                for j in range(d):
                    for k in range(Nnodes):
                        HX[i,k,j]=H[j,i,k].X
                        EtHX[i,k,j]=EtH[j,i,k].X
                        HtEX[i,k,j]=HtE[j,i,k].X
                    for k in range(NEedges):
                        P_edgeX[i,k,j]=P_edge[j,i,k].X
                    for k in range(NHedges):
                        H_edgeX[i,k,j]=H_edge[j,i,k].X
                
    if long_outs == False:
        return outputs
    else:
        return outputs, HX, EtHX, HtEX, P_edgeX,H_edgeX

#%% OPT 3 - network

def OPT3(network):
    """
    Basically OPT2 but without grouping over scenarios.
    """
    if network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly

    if "node" in network.n.columns:
        network.n.set_index("node", inplace=True)   

    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d_loadP = network.loadP_t.shape[2] #number of scenarios for demand
    d = network.n_scenarios #number of scenarios
    inst = network.loadP_t.shape[0] #number of time steps T


    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)

    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])    
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])   
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC) 
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH) 

    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ch_t/d, lb=0)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge/d, lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge/d, lb=0)
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( P_edge[j,i,k] >= P_edge[j,i,k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= H_edge[j,i,k] for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -P_edge[j,i,k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -H_edge[j,i,k] for i in range(inst) for j in range(d) for k in range(NHedges))

    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes)) 
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( -P_edge[j,i,k] <= network.edgesP['NTC'].iloc[k] + addNTC[k] for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( -H_edge[j,i,k] <= network.edgesH['MH'].iloc[k] + addMH[k] for i in range(inst) for j in range(d) for k in range(NHedges))


    
    #todo: sobstitute 30 with a parameter
    cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -  
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
                            quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            ==0 for j in range(d) for k in range(Nnodes))
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')



    ES = network.genS_t
    EW = network.genW_t
    EL = network.loadP_t
    HL = network.loadH_t

    if d_loadP == 1:
        for j in range(d): 
            for k in range(Nnodes):
                for i in range(inst-1):
                    cons2[j,i,k].rhs = HL[i,k,0] #time,node,scenario or if you prefer to not remember use isel     
                cons3[j,k].rhs  = HL[inst-1,k,0]
        cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list()) 
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))

    else:
        for j in range(d): 
            for k in range(Nnodes):
                for i in range(inst-1):
                    cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel     
                cons3[j,k].rhs  = HL[inst-1,k,j]
        cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list()) 
                            >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst))


    model.optimize()

    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        node_dims = ["scenario","time","node"]
        node_coords = [ range(d), range(inst),  network.n.index.to_list()]
        edge_dims = ["scenario","time","edge"]
        edge_coords = [ range(d),  range(inst), range(NEedges)]
    

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
        }

        #print("opt time: {}s.".format(np.round(time.time()-start_time,3)))
        return [VARS]



# %% OPT_agg


def OPT_agg(network):
    """
	Performs optimization on a given network.
    The network has an initialized time partition to aggregate the data and variables approriately.
    The aggregated model is then solved. The aggregated model is a relaxation of the original model (solved in OPT3).

	Parameters:
	- network: an instance of the network class.

	Returns:
	- outputs: a list containing the optimized variables and the objective value.



	"""
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly


    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = network.n_scenarios 
    inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    tp_obj = network.time_partition
    tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')

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

    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,obj=cP_edge/d) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,obj=cH_edge/d)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))

    outputs=[]
    VARS=[]
    #todo perchè lo rimuoviamo questo?

    #todo: sobstitute 30 with a parameter
    # cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
    #                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
    #                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
    #                        ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    # cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
    #                        quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
    #                        quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
    #                        ==0 for j in range(d) for k in range(Nnodes))

    ES = network.genS_t_agg
    EW = network.genW_t_agg
    EL = network.loadP_t_agg
    HL = network.loadH_t_agg

   
    if network.loadP_t_agg.shape[2] > 1:
        cons2=model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
        # cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
        #                     quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[inst-1,k,j] for j in range(d) for k in range(Nnodes))
    # for j in range(d):
        #     for k in range(Nnodes):
        #         for i in range(inst-1):
                    #cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel
                #cons3[j,k].rhs  = HL[inst-1,k,j]
        
        cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
    
    else:
        cons2=model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(inst) for k in range(Nnodes)))
        # cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
        #                     quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[inst-1,k,j] for j in range(d) for k in range(Nnodes))
        # for j in range(d):
        #     for k in range(Nnodes):
        #         for i in range(inst-1):
        #             cons2[j,i,k].rhs = HL[i,k,0] #time,node,scenario or if you prefer to not remember use isel
        #         cons3[j,k].rhs  = HL[inst-1,k,0]

        cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst)))
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        node_dims = ["scenario","time","node"]
        if 'node' in network.n.columns:
            network.n.set_index('node',inplace=True)
        node_coords = [ range(d), range(inst),  network.n.index.to_list()]
        edge_dims = ["scenario","time","edge"]
        edge_coords = [ range(d),  range(inst), range(NEedges)]
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
            "interval_to_var":dict(zip(time_partition.tuplize(tp),range(inst))),
            "var_to_interval":dict(zip(range(inst),time_partition.tuplize(tp)))  
        }

                                                   

        #print("opt time: {}s.".format(np.round(time.time()-start_time,3)))

    return [VARS]#HH,ETH,HTE




#%% OPT_agg_correct

def OPT_agg_correct(network):
    """
	Performs optimization on a given network.
    The network has an initialized time partition to aggregate the data and variables approriately.
    The aggregated model is then solved. The aggregated model is a relaxation of the original model (solved in OPT3).

	Parameters:
	- network: an instance of the network class.

	Returns:
	- outputs: a list containing the optimized variables and the objective value.

	"""
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly


    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = network.n_scenarios 
    inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    tp_obj = network.time_partition
    tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')

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

    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge/d, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge/d, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge/d, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge/d, lb=0)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))

    VARS=[]

    ES = network.genS_t_agg
    EW = network.genW_t_agg
    EL = network.loadP_t_agg
    HL = network.loadH_t_agg

   
    if network.loadP_t_agg.shape[2] > 1:
        model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))

        
        model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
    
    else:
        model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(inst) for k in range(Nnodes)))


        model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst)))
        
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        node_dims = ["scenario","time","node"]
        if 'node' in network.n.columns:
            network.n.set_index('node',inplace=True)
        node_coords = [ range(d), range(inst),  network.n.index.to_list()]
        edge_dims = ["scenario","time","edge"]
        edge_coords = [ range(d),  range(inst), range(NEedges)]
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
            "P_edge_pos":solution_to_xarray(P_edge_pos, edge_dims, edge_coords),
            "H_edge_pos":solution_to_xarray(H_edge_pos, edge_dims, edge_coords),
            "P_edge_neg":solution_to_xarray(P_edge_neg, edge_dims, edge_coords),
            "H_edge_neg":solution_to_xarray(H_edge_neg, edge_dims, edge_coords),
            "HtE":solution_to_xarray(HtE, node_dims, node_coords),
            "obj":model.ObjVal,
            "interval_to_var":dict(zip(time_partition.tuplize(tp),range(inst))),
            "var_to_interval":dict(zip(range(inst),time_partition.tuplize(tp)))  
        }

                                                   

        print("opt time: {}s.".format(np.round(time.time()-start_time,3)))

    return [VARS]#HH,ETH,HTE




#%%

#OPT_agg2(network):
#  """
# 	Performs optimization on a given network.
#     The network has an initialized time partition to aggregate the data and variables approriately.
#     The aggregated model is then solved. The aggregated model is a relaxation of the original model (solved in OPT3).

# 	Parameters:
# 	- network: an instance of the network class.

# 	Returns:
# 	- outputs: a list containing the optimized variables and the objective value.



#     return [VARS]#HH,ETH,HTE
# """

# # %% 
# define network 

# eu = EU()
# eu.init_time_partition()
#notagg_results = OPT3(eu)
# T =  eu.genW_t.shape[0]
# tp_obj = time_partition(T)
# eu.init_time_partition()
# eu.update_time_partition()
# results = []
# #%%

# iter_start_time=time.time()
# for i in range(40):
#     agg_results = OPT_agg(eu)
#     results += [agg_results]
#     eu.iter_partition(5)


# #print('end iter in {}s'.format(np.round(time.time()-iter_start_time),3))
# #%%
# results_notagg = OPT3(eu)
# %% OPT_time_partition

#input parameters
# network = eu
# N_iter = 10
# N_refining = 4
#function code

def OPT_time_partition(network, N_iter = 10, N_refining = 4):
        
    def tuplize(tp):
        l = []
        for i in tp:
            if type(i) is list:
                l.append(tuple(i))
            else:
                l.append(i)
        return l

    def invert_dict(d):
        """
        Inverts a dictionary by swapping the keys and values.

        Parameters:
            d (dict): The dictionary to be inverted.

        Returns:
            dict: The inverted dictionary where the values of the original dictionary are the keys and the keys of the original dictionary are the values. If multiple keys in the original dictionary have the same value, the values in the inverted dictionary will be a list of keys.

        Example:
            >>> d = {'a': 1, 'b': 2, 'c': 1}
            >>> invert_dict(d)
            {1: ['a', 'c'], 2: ['b']}
        """
        inverted = {}
        for key, value in d.items():
            if value not in inverted:
                inverted[value] = [key]
            else:
                inverted[value].append(key)
        return inverted


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

    def add_H_balance_constraint(i0,i1,name = None):
        """
        Adds constraints to the model to ensure the balance of H variables.

        Args:
            i0 (int): time step index.
            i1 (int): "next" time step index.

        Note:
            - The function assumes that the `model`, `H`, `network`, `EtH`, `H_edge`, `HL`, `var_to_interval_index`, and `Nnodes` variables are defined and accessible in the scope of the function.
            - The function does not return any value.
        """

        if name is None: 
            if HL.shape[2] == 1:
                model.addConstrs((- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                            quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            ==HL[var_to_interval_index[i0],k,0]  for j in range(d) for k in range(Nnodes)))
            else:
                model.addConstrs((- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                ==HL[var_to_interval_index[i0],k,j]  for j in range(d) for k in range(Nnodes)))
            
        else:
            if HL.shape[2] == 1:
                model.addConstrs((- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                ==HL[var_to_interval_index[i0],k,0] for j in range(d) for k in range(Nnodes)), name = name)
                # #print("#print to debug:",HtE[0,i0,0], H_edge[0,i0,0],   quicksum(H_edge[0,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[0]].index.to_list()),var_to_interval_index[i0], HL.shape)
                # contrs = model.addConstr((- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                #         quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                #         quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                #         ==HL[var_to_interval_index[i0],k,0] for j in range(d) for k in range(Nnodes)), name = name)
            else:
                model.addConstrs((- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                ==HL[var_to_interval_index[i0],k,j] for j in range(d) for k in range(Nnodes)), name = name)

    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

    if network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, ch_t, chte, ceth, cNTC, cMH = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly

    if "node" in network.n.columns:
        network.n.set_index("node", inplace=True)

    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = network.n_scenarios #number of scenarios
    inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    tp_obj = network.time_partition
    tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')

    #tracking relation between variables and time partition
    var_to_interval = dict(zip(range(inst),tp)) #dictionary saying which interval each variable time intex rapresents
    var_to_partition = dict(zip(range(inst),[0]*inst)) #dictionary saying which partition each variable time index was added from
    var_to_interval_index = dict(zip(range(inst),range(inst))) #maps var to the index of the interval in the corresponding time partition
    interval_to_var = dict(zip(tuplize(tp),range(inst)))


    env = Env(params={'OutputFlag': 0}) 
    model = Model(env=env)
    #model.setParam('LPWarmStart',2)
    #model.setParam('Method',1)
    #time and scenario independent variables
    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)

    #time and scenario dependent variables
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))


    #cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
    #todo: sobstitute 30 with a parameter
    #TODO non è che il motivo per cui viene unfeasible è lo stesso per cui non ha senso se metto in constraint qui?
   
    
    ES = network.genS_t_agg
    EW = network.genW_t_agg
    EL = network.loadP_t_agg
    HL = network.loadH_t_agg

    #model.remove(cons1)
    # if HL.shape[2] == 1: #then we only have one scenario, use it for all.
    #     for j in range(d):
    #         for k in range(Nnodes):
    #             for i in range(inst-1):
    #                 cons2[j,i,k].rhs = HL[i,k,0] #time,node,scenario or if you prefer to not remember use isel
    #             cons3[j,k].rhs  = HL[inst-1,k,0] #per qualche motivo se aggiungo questo direttamente mi da unfeasible???
    # else:
    #      for j in range(d):
    #         for k in range(Nnodes):
    #             for i in range(inst-1):
    #                 cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel
    #             cons3[j,k].rhs  = HL[inst-1,k,j] #per qualche motivo se aggiungo questo direttamente mi da unfeasible???


    if EL.shape[2] == 1: #then we only have one scenario, use it for all.
        model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            == HL[i,k,0] for j in range(d) for i in range(inst-1) for k in range(Nnodes)), name ='hydrogen_balance_0')
       
        model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                        >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst)), name="power_balance_0")
    else:
        model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            == HL[i,k,j] for j in range(d) for i in range(inst-1) for k in range(Nnodes)), name ='hydrogen_balance_0')

        model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)), name="power_balance_0")

    
    outputs=[]
    VARS=[]
    #print('first OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')


    model.optimize()
    if model.Status!=2:
        #print("Status = {}".format(model.Status))
        model.computeIIS()
        constrs = model.getConstrs()
        IIS = []
        for c in constrs:  
            if c.IISConstr:
                IIS.append(c)
        print(IIS)
    else:
        VARS =[[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]]
        outputs= outputs + [VARS+[model.ObjVal]]
        print("opt time: {}s.".format(np.round(time.time()-start_time,3)))


    #now we try to refine partition and add variables and constraints appropriately.abs
    for partition_generation in range(1,N_iter): #se si itera più di una volta fallisce, rip.
        generation_start_time = time.time()
        #iter over partition_generation
        network.iter_partition(N_refining)
        print("iter_partition in {}s".format(np.round(time.time()-generation_start_time,3)))
        family_tree = network.time_partition.family_tree
        splitted_intervals = order_intervals(family_tree[-1])
        ##print(splitted_intervals)
        tp_obj = network.time_partition #new time partition object
        tp = tp_obj.agg

        is_not_first_round = False

        for father_interval in splitted_intervals:
            split_indeces = interval_subsets(father_interval,tp) #indexes of tp that are subsets of the father_interval
            father_index = interval_to_var[tuple(father_interval)]
            
            # update dioctionaries do I really need them all
            new_vars = range(len(var_to_interval),len(var_to_interval)+len(split_indeces))
            interval_index_to_var = dict(zip(split_indeces,new_vars)) #temporary dictionary: tp position to corresponding variable
            var_to_interval = {**var_to_interval, **dict(zip(new_vars,[tp[i] for i in split_indeces]))} #dictionary saying which interval each variable time intex rapresents
            var_to_partition = {**var_to_partition, **dict(zip(new_vars,[partition_generation]*len(new_vars)))} #dictionary saying which partition each variable time index was added from
            var_to_interval_index = {**var_to_interval_index,**dict(zip(new_vars,split_indeces))} #maps var to the index of the interval in the corresponding time partition
            interval_to_var = {**interval_to_var,**dict(zip(tuplize([tp[i] for i in split_indeces]),new_vars))} #ATTENTION KEYS ARE TUPLES!!!

            #add new variables
            HtE = {**HtE, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0)} # expressed in kg
            EtH = {**EtH, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0)} # expressed in MWh
            H = {**H, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)}
            P_edge = {**P_edge, **model.addVars(product(range(d),new_vars,range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)} #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
            H_edge = {**H_edge, **model.addVars(product(range(d),new_vars,range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)}

            #constraint binding the right interval of the newly split interval to the previous time step
            if is_not_first_round:
                outer_right = interval_to_var[tuple_index]
                add_H_balance_constraint(inner_right,outer_right, name=f'right-{inner_right}-H_balance_{partition_generation}')
                ##print("Not first round, adding constraints for right extreme of interval {}{}".format(inner_right,outer_right))
            #modulo constriants
            model.addConstrs( (H[j,i,k] <= nh[k] for i in new_vars for j in range(d) for k in range(Nnodes)), name = f'max_H_{partition_generation}')
            model.addConstrs( (EtH[j,i,k] <= meth[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes)), name = f'max_EtH_{partition_generation}')
            model.addConstrs( (HtE[j,i,k] <= mhte[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes)), name = f"max_HtE_{partition_generation}")
            model.addConstrs( (P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges)), name = f"max_P_edge_{partition_generation}")
            model.addConstrs( (H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges)), name = f"max_H_edge_{partition_generation}")
            model.addConstrs( (P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges)), name = f"min_P_edge_{partition_generation}")
            model.addConstrs( (H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges)), name = f"min_H_edge_{partition_generation}")

            #relation between new variables and old variables (i am actually unsure whethere these are necessary)
            #model.addConstrs((EtH[j, father_index,k] == quicksum(EtH[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes)), name = f"father_son_EtH_{partition_generation}")
            #model.addConstrs((HtE[j, father_index,k] == quicksum(HtE[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes)), name = f"father_son_HtE_{partition_generation}")
            #model.addConstrs((P_edge[j, father_index,k] == quicksum(P_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NEedges)), name = f"father_son_P_edge_{partition_generation}")
            #model.addConstrs((H_edge[j, father_index,k] == quicksum(H_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NHedges)), name = f"father_son_H_edge_{partition_generation}")
            #model.addConstrs((H[j,father_index, k] == H[j, interval_index_to_var[split_indeces[-1]],k] for j in range(d) for k in range(Nnodes))) #se metto split_indexes[0] diventa unfeasible umpf
            # # new aggregated scenarios
            ES = network.genS_t_agg
            EW = network.genW_t_agg
            EL = network.loadP_t_agg
            HL = network.loadH_t_agg

            #Power balance constraints
            if EL.shape[2] == 1:
                model.addConstrs((ns[k]*ES[var_to_interval_index[i],k,j] + nw[k]*EW[var_to_interval_index[i],k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                >= EL[var_to_interval_index[i],k,0] for k in range(Nnodes) for j in range(d) for i in new_vars), name=f"power_balance_{partition_generation}")
            else:
                model.addConstrs((ns[k]*ES[var_to_interval_index[i],k,j] + nw[k]*EW[var_to_interval_index[i],k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                        >= EL[var_to_interval_index[i],k,j] for k in range(Nnodes) for j in range(d) for i in new_vars), name=f"power_balance_{partition_generation}")

            #Complicating constraints
            for i in new_vars[:-1]:
                add_H_balance_constraint(i,i+1, name = f"{i}-H_balance_{partition_generation}")

            #todo con2 for time step before the first: look at previous interval in tp if it's a new interval look at the corresponding variable if it isn't, look at the corresponding variable in the partition in which it was added.abs
            #stessa cosa per l'ultimo time step.


            #constraint binding the left interval of the newly split interval to the previous time step
            left_interval = tp[split_indeces[0]-1]
            if type(left_interval) is list: 
                ##print("left_interval is list",left_interval)
                tuple_index = tuple(left_interval)
            else:
                ##print("left interval is not list",left_interval)
                tuple_index = left_interval

            outer_left = interval_to_var[tuple_index]
            inner_left = interval_index_to_var[split_indeces[0]]
            ##print(f"{outer_left}-{inner_left} var binded")
            add_H_balance_constraint(outer_left,inner_left, name = f"left-{inner_left}-H_balance_{partition_generation}")
            

            if split_indeces[-1] != len(tp)-1:
                right_interval = tp[split_indeces[-1]+1] #questo non da out of index perchè l'ultimo caso sta procedendo in ordine

                if type(right_interval) is list: 
                    ##print("right_interval is list",right_interval)
                    tuple_index = tuple(right_interval)
                else:
                    ##print("right_interval is not list",left_interval)
                    tuple_index = right_interval

            
                inner_right = interval_index_to_var[split_indeces[-1]]
            

            is_not_first_round = True

            #ora devo solo trovare le variabili corrispondenti e poi è fatta yee, ma right_interval potrebbe non avere ancora variabili associate umpf umpf
        last_interval = splitted_intervals[-1]

        #update final balance constraint if necessary,otherwise add right interval constraint
        # TODO: se la prima partizione contiene zero?                         
        if last_interval[-1] == network.T-1:
            ##print("last interval contains T")
            i1 = interval_to_var[tuplize(tp)[0]] #variable corresponding to the first interval
            i0 = interval_to_var[tp[-1]]
            add_H_balance_constraint(i0,i1)
        else:
            ##print("last interval doesn't contain T")
            right_interval = tp[split_indeces[-1]+1]
            if type(right_interval) is list: 
                tuple_index = tuple(right_interval)
            else:
                tuple_index = right_interval
            i0 = interval_index_to_var[split_indeces[-1]]
            i1 = interval_to_var[tuple_index]
            add_H_balance_constraint(i0,i1,name = f"right-{i0}-H_balance_{partition_generation}")
            

        #print(f"OPT Model redefinition iteration {partition_generation} time: {np.round(time.time()-generation_start_time,3)}s.")
        opt_start_time = time.time()
        model.optimize()
        
        if model.Status!=2:
            #print("Unfesasible or Unbounded Status = {}".format(model.Status))
            model.computeIIS()
            constrs = model.getConstrs()
            IIS = []
            for c in constrs:  
                if c.IISConstr:
                    IIS.append(c)
                    
            #print(IIS)
            #print("returning iis")
            return IIS
            # IIS
            # if model.status == GRB.INFEASIBLE:
            #     #print("relaxing constraints...")
            #     model.feasRelaxS(1, False, False, True)
            #     model.optimize()
            # VARS  =[[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]]
            # outputs=outputs + [VARS+[model.ObjVal]]
            # #print("opt time: {}s.".format(np.round(time.time()-start_time,3)))
        else:
            if partition_generation == N_iter-1:
                #print("Last iteration done, saving last variables")
                node_dims = ["scenario","time","node"]
                node_coords = [ range(d), range(len(var_to_interval)),  network.n.index.to_list()]
                edge_dims = ["scenario","time","edge"]
                edge_coords = [ range(d), range(len(var_to_interval)), range(NEedges)]
                VARS=dict(zip(["ns","nw","nh","mhte","meth","H","EtH","P_edge","H_edge","HtE","obj","interval_to_var","var_to_interval"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                            solution_to_xarray(H, node_dims, node_coords), 
                                                            solution_to_xarray(EtH, node_dims, node_coords), 
                                                            solution_to_xarray(P_edge, edge_dims, edge_coords), 
                                                            solution_to_xarray(H_edge, edge_dims, edge_coords), 
                                                            solution_to_xarray(HtE, node_dims, node_coords),
                                                            model.ObjVal, interval_to_var, var_to_interval]))
            else:
                #print("iteration done, saving variables")
                VARS=dict(zip(["ns","nw","nh","mhte","meth","obj"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                                model.ObjVal]))
            outputs= outputs + [VARS]
            #print("opt time: {}s.".format(np.round(time.time()-opt_start_time,3)))
    #print("total optimization time: {}s.".format(np.round(time.time()-start_time,3)))
    if N_iter == 0:
            node_dims = ["scenario","time","node"]
            node_coords = [ range(d), range(len(var_to_interval)),  network.n.index.to_list()]
            edge_dims = ["scenario","time","edge"]
            edge_coords = [ range(d), range(len(var_to_interval)), range(NEedges)]
            VARS=dict(zip(["ns","nw","nh","mhte","meth","H","EtH","P_edge","H_edge","HtE","obj","interval_to_var","var_to_interval"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                        solution_to_xarray(H, node_dims, node_coords), 
                                                        solution_to_xarray(EtH, node_dims, node_coords), 
                                                        solution_to_xarray(P_edge, edge_dims, edge_coords), 
                                                        solution_to_xarray(H_edge, edge_dims, edge_coords), 
                                                        solution_to_xarray(HtE, node_dims, node_coords),
                                                        model.ObjVal, interval_to_var, var_to_interval]))
       
    
            outputs = outputs + [VARS]
    #network.results = outputs
    return outputs


#%% OPT_time_partition_old:

def OPT_time_partition_old(network, N_iter = 5, N_refining = 1):
    def tuplize(tp):
        l = []
        for i in tp:
            if type(i) is list:
                l.append(tuple(i))
            else:
                l.append(i)
        return l
    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

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


    def add_H_balance_constraint(i0,i1):
        model.addConstrs(- H[j,i1,k] + H[j,i0,k] + 30*network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                                    quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(H_edge[j,i0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    ==HL[var_to_interval_index[i0],k,j] for j in range(d) for k in range(Nnodes))


    if network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly


    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = network.loadP_t_agg.shape[2] #number of scenarios
    inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    tp_obj = network.time_partition
    tp = tp_obj.agg #time partition
    #print(f'sanity checl, is inst equal to len tp= {inst == len(tp)}')

    #tracking relation between variables and time partition
    var_to_interval = dict(zip(range(inst),tp)) #dictionary saying which interval each variable time intex rapresents
    var_to_partition = dict(zip(range(inst),[0]*inst)) #dictionary saying which partition each variable time index was added from
    var_to_interval_index = dict(zip(range(inst),range(inst))) #maps var to the index of the interval in the corresponding time partition
    interval_to_var = dict(zip(tuplize(tp),range(inst)))


    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario independent variables
    ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)

    #time and scenario dependent variables
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    #fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))


    #cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
    #todo: sobstitute 30 with a parameter

    cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
                            quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(H_edge[j,inst-1,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                            ==0 for j in range(d) for k in range(Nnodes))
    #print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')


    ES = network.genS_t_agg
    EW = network.genW_t_agg
    EL = network.loadP_t_agg
    HL = network.loadH_t_agg

    #model.remove(cons1)
    for j in range(d):
        for k in range(Nnodes):
            for i in range(inst-1):
                cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel
            cons3[j,k].rhs  = HL[inst-1,k,j]


    cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                        >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst))
    outputs=[]
    VARS=[]

    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        VARS+=[[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]]
        outputs=outputs + [VARS+[model.ObjVal]]
        print("opt time: {}s.".format(np.round(time.time()-start_time,3)))

    #now we try to refine partition and add variables and constraints appropriately.abs
    for partition_generation in range(N_iter):
        #iter over partition_generation
        network.iter_partition(N_refining)


        # how to append variables len({**H , **H[list(H.keys())[-2]]})
        # set(a).issubset(b) to check subset relation in lists can be made faster in the class though
        family_tree = network.time_partition.family_tree

        def order_intervals(L):
            return sorted(L, key=lambda l: l[0])

        splitted_intervals = order_intervals(family_tree[-1])
        tp_obj = network.time_partition #new time partition object
        tp = tp_obj.agg

        is_not_first_round = False

        for father_interval in splitted_intervals:
            split_indeces = interval_subsets(father_interval,tp)
            father_index = interval_to_var[tuple(father_interval)]
            
            # update dioctionaries do I really need them all
            new_vars = range(len(var_to_interval),len(var_to_interval)+len(split_indeces))
            interval_index_to_var = dict(zip(split_indeces,new_vars)) #temporary dictionary: tp position to corresponding variable
            var_to_interval = {**var_to_interval, **dict(zip(new_vars,[tp[i] for i in split_indeces]))} #dictionary saying which interval each variable time intex rapresents
            var_to_partition = {**var_to_partition, **dict(zip(new_vars,[partition_generation]*len(new_vars)))} #dictionary saying which partition each variable time index was added from
            var_to_interval_index = {**var_to_interval_index,**dict(zip(new_vars,split_indeces))} #maps var to the index of the interval in the corresponding time partition
            interval_to_var = {**interval_to_var,**dict(zip(tuplize([tp[i] for i in split_indeces]),new_vars))} #ATTENTION KEYS ARE TUPLES!!!

            #add new variables
            HtE = {**HtE, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0)} # expressed in kg
            EtH = {**EtH, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0)} # expressed in MWh
            H = {**H, **model.addVars(product(range(d),new_vars,range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)}
            P_edge = {**P_edge, **model.addVars(product(range(d),new_vars,range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)} #could make sense to sosbstitute Nodes with network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
            H_edge = {**H_edge, **model.addVars(product(range(d),new_vars,range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)}

            #constraint binding the right interval of the newly split interval to the previous time step
            if is_not_first_round:
                outer_right = interval_to_var[tuple_index]
                add_H_balance_constraint(inner_right,outer_right)

            #modulo constriants
            model.addConstrs( H[j,i,k] <= nh[k] for i in new_vars for j in range(d) for k in range(Nnodes))
            model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes))
            model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes))
            model.addConstrs( P_edge[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges))
            model.addConstrs( H_edge[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges))
            model.addConstrs( P_edge[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges))
            model.addConstrs( H_edge[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges))

            #relation between new variables and old variables (i am actually unsure whethere these are necessary)
            model.addConstrs(EtH[j, father_index,k] == quicksum(EtH[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes))
            model.addConstrs(HtE[j, father_index,k] == quicksum(HtE[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes))
            #model.addConstrs(P_edge[j, father_index,k] == quicksum(P_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NEedges))
            #model.addConstrs(H_edge[j, father_index,k] == quicksum(H_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NHedges))
            #model.addConstrs(H[j,father_index, k] == H[j, interval_index_to_var[split_indeces[-1]],k] for j in range(d) for k in range(Nnodes))
            # # new aggregated scenarios
            ES = network.genS_t_agg
            EW = network.genW_t_agg
            EL = network.loadP_t_agg
            HL = network.loadH_t_agg

            #Power balance constraints
            model.addConstrs(ns[k]*ES[var_to_interval_index[i],k,j] + nw[k]*EW[var_to_interval_index[i],k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[var_to_interval_index[i],k,j] for k in range(Nnodes) for j in range(d) for i in new_vars)

            #Complicating constraints
            add_H_balance_constraint(i+1,i)

            #todo con2 for time step before the first: look at previous interval in tp if it's a new interval look at the corresponding variable if it isn't, look at the corresponding variable in the partition in which it was added.abs
            #stessa cosa per l'ultimo time step.


            #constraint binding the left interval of the newly split interval to the previous time step
            left_interval = tp[split_indeces[0]-1]
            if type(left_interval) is list: 
                tuple_index = tuple(left_interval)
            else:
                tuple_index = left_interval

            outer_left = interval_to_var[tuple_index]
            inner_left = interval_index_to_var[split_indeces[0]]
            
            add_H_balance_constraint(outer_left,inner_left)
            

            
            right_interval = tp[split_indeces[-1]+1]

            if type(right_interval) is list: 
                tuple_index = tuple(right_interval)
            else:
                tuple_index = right_interval

        
            inner_right = interval_index_to_var[split_indeces[-1]]
        

            is_not_first_round = True

            #ora devo solo trovare le variabili corrispondenti e poi è fatta yee, ma right_interval potrebbe non avere ancora variabili associate umpf umpf

            #TODO
            #todo vvvvv: forse non va messo? solo se ultimo timespet è T se si mette che l'ultimo time step della nuova partizioe del livello di idrogeno corrisponde con quella prec ma inst-1 cambia mhh ok

        last_interval = splitted_intervals[-1]

        #update final balance constraint if necessary,otherwise add right interval constraint                         
        if last_interval[-1] == network.T:
            i1 = interval_to_var[tuplize(tp)[0]] #variable corresponding to the first interval
            add_H_balance_constraint(interval_index_to_var[len(tp)],i1)
        else:
            right_interval = tp[split_indeces[-1]+1]
            if type(right_interval) is list: 
                tuple_index = tuple(right_interval)
            else:
                tuple_index = right_interval
            i0 = interval_index_to_var[split_indeces[-1]]
            i1 = interval_to_var[tuple_index]
            add_H_balance_constraint(i0,i1)
            


        opt_start_time = time.time()
        model.optimize()
          
        if model.Status!=2:
            print("Unfesasible or Unbounded Status = {}".format(model.Status))
            model.computeIIS()
            constrs = model.getConstrs()
            IIS = []
            for c in constrs:  
                if c.IISConstr:
                    IIS.append(c)
                    
            print(IIS)
            print("returning iis")
            return IIS
            # IIS
            # if model.status == GRB.INFEASIBLE:
            #     #print("relaxing constraints...")
            #     model.feasRelaxS(1, False, False, True)
            #     model.optimize()
            # VARS  =[[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]]
            # outputs=outputs + [VARS+[model.ObjVal]]
            # #print("opt time: {}s.".format(np.round(time.time()-start_time,3)))
        else:
            if partition_generation == N_iter-1:
                #print("Last iteration done, saving last variables")
                node_dims = ["scenario","time","node"]
                node_coords = [ range(d), range(len(var_to_interval)),  network.n.index.to_list()]
                edge_dims = ["scenario","time","edge"]
                edge_coords = [ range(d), range(len(var_to_interval)), range(NEedges)]
                VARS=dict(zip(["ns","nw","nh","mhte","meth","H","EtH","P_edge","H_edge","HtE","obj","interval_to_var","var_to_interval"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                            solution_to_xarray(H, node_dims, node_coords), 
                                                            solution_to_xarray(EtH, node_dims, node_coords), 
                                                            solution_to_xarray(P_edge, edge_dims, edge_coords), 
                                                            solution_to_xarray(H_edge, edge_dims, edge_coords), 
                                                            solution_to_xarray(HtE, node_dims, node_coords),
                                                            model.ObjVal, interval_to_var, var_to_interval]))
            else:
                #print("iteration done, saving variables")
                VARS=dict(zip(["ns","nw","nh","mhte","meth","obj"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                                model.ObjVal]))
            outputs= outputs + [VARS]
            #print("opt time: {}s.".format(np.round(time.time()-opt_start_time,3)))
    #print("total optimization time: {}s.".format(np.round(time.time()-start_time,3)))
    if N_iter == 0:
            node_dims = ["scenario","time","node"]
            node_coords = [ range(d), range(len(var_to_interval)),  network.n.index.to_list()]
            edge_dims = ["scenario","time","edge"]
            edge_coords = [ range(d), range(len(var_to_interval)), range(NEedges)]
            VARS=dict(zip(["ns","nw","nh","mhte","meth","H","EtH","P_edge","H_edge","HtE","obj","interval_to_var","var_to_interval"],[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)]),
                                                        solution_to_xarray(H, node_dims, node_coords), 
                                                        solution_to_xarray(EtH, node_dims, node_coords), 
                                                        solution_to_xarray(P_edge, edge_dims, edge_coords), 
                                                        solution_to_xarray(H_edge, edge_dims, edge_coords), 
                                                        solution_to_xarray(HtE, node_dims, node_coords),
                                                        model.ObjVal, interval_to_var, var_to_interval]))
       
    
            outputs = [VARS]

    return outputs
#%% OPT_agg2

def OPT_agg2(network, N_iter, iter_method = "random", k = 1):
    """
    This function solves the optimization problem using the Gurobi solver. It takes as input a network object and the number of iterations for the optimization. It returns a list of dictionaries, where each dictionary contains the solution of the optimization problem at each iteration.

    The function first sets up the optimization problem by defining the variables, constraints and objective function. Then it iteratively solves the optimization problem for each iteration of the time partition. It adds constraints to the model at each iteration by using the addConstrs method of the Gurobi model class. The function also #prints the total optimization time and the time for each iteration.

    The function is useful for solving the optimization problem for a given network for different time partitions. The output of the function can be used to analyze the results of the optimization problem for different time partitions.

    The function returns a list of dictionaries, where each dictionary contains the solution of the optimization problem at each iteration. The dictionary contains the following keys: 'ns', 'nw', 'nh', 'mhte', 'meth', 'addNTC', 'addMH', 'H', 'EtH', 'P_edge', 'H_edge', 'HtE', 'obj', 'interval_to_var' and 'var_to_interval'. The values of the keys are the solution of the optimization problem at each iteration.
    The function #prints the total optimization time and the time for each iteration. The function also #prints the status of the optimization problem at each iteration.

    """
    if network.costs.shape[0] == 1: #if the costs are the same:
         cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
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
    #TODO: fai due grafi diversi
    H_edge = model.addVars(product(range(d),range(T),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)
    P_edge_pos = model.addVars(product(range(d),range(T),range(NEedges)),vtype=GRB.CONTINUOUS,lb=0, obj = cP_edge/d)
    H_edge_pos = model.addVars(product(range(d),range(T),range(NHedges)),vtype=GRB.CONTINUOUS,lb=0, obj = cH_edge/d)
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( P_edge_pos[j,i,k] >= P_edge[j,i,k] for i in range(T) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] >= H_edge[j,i,k] for i in range(T) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] >= -P_edge[j,i,k] for i in range(T) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] >= -H_edge[j,i,k] for i in range(T) for j in range(d) for k in range(NHedges))
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
                            >=  EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(Ntp)))

    else:
        cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(Ntp) for k in range(Nnodes)))

        cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j]  + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                            quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >=  EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(Ntp)))
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
    #%  return VARS
    for iter in range(N_iter):
        iter_start_time = time.time()

        VARS = iter_sol[-1]
        if iter_method == "random":
            print("random iteration")
            network.iter_partition(k=k)
        elif iter_method == "rho":
            print("rho iteration")
            network.rho_iter_partition(VARS, k=k)
        elif iter_method == "validation":
            print("validation iteration")
            optimal = network.validationfun_iter_partition(VARS, k=k)

            if optimal:
                print("optimal solution for unaggregated problem found")
                return iter_sol
        else:
            raise ValueError("Invalid iteration method.")

        family_tree = network.time_partition.family_tree
        splitted_intervals = time_partition.order_intervals(family_tree[-1])
        ##print(splitted_intervals)
        tp_obj = network.time_partition #new time partition object
        tp = tp_obj.agg.copy()
        tp = [t if type(t) is list else [t] for t in tp]
        Ntp = len(tp)
        ES = network.genS_t_agg
        EW = network.genW_t_agg
        EL = network.loadP_t_agg
        HL = network.loadH_t_agg


        for father_interval in splitted_intervals:
            #print(f"father interval: {father_interval} \n len tp:{len(tp)} \n shape HL: {HL.shape}")
            
            split_indeces = time_partition.interval_subsets(father_interval,tp)[1:] #indexes of tp that are subsets of the father_interval, except the first one sine it's implied by linear dependence of the corresponding contraints.
            #print("split indeces: {}".format(split_indeces))
            NI = len(split_indeces)
            #print
            if network.loadP_t_agg.shape[2] > 1:
                #TODO: sarebbe bello calcolare la domanda sull0intervallo direttamente qui e non usare i dataframe aggregati (per essere ricuri di non fare casino)
                cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                                quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL[i,k,j] for j in range(d) for i in split_indeces for k in range(Nnodes)))
                
                cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                                    quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in split_indeces))

            else:
                cons2=model.addConstrs((- H[j,tp[(i+1)%Ntp][0],k] + H[j,tp[i][0],k] + 30*network.n['feth'].iloc[k]*quicksum(EtH[j,t,k] for t in tp[i]) - quicksum(HtE[j,t,k] for t in tp[i]) -
                                quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,t,l] for t in tp[i] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL[i,k,0] for j in range(d) for i in split_indeces for k in range(Nnodes)))

                cons1=model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j]  + 0.033*network.n['fhte'].iloc[k]*quicksum(HtE[j,t,k] for t in tp[i]) - quicksum(EtH[j,t,k] for t in tp[i]) -
                                    quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                    quicksum(P_edge[j,t,l] for t in tp[i] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    >= EL[i,k,0] for j in range(d) for i in split_indeces for k in range(Nnodes)))
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
                "var_to_interval":dict(zip(range(T),time_partition.tuplize(tp))),
                "opt_time":np.round(time.time()-start_time,3) 
            }
            iter_sol.append(VARS)
        

    print(f"Total opt time: {np.round(time.time()-start_time,3)}s.")
   
    return iter_sol
        
#%%
#nnode_to_node = dict(zip(range(Nnodes),network.n['node'].to_list()))
# if __name__ == "__main__":
#     time_coords = list(var_to_interval.keys())
#     H_sol = solution_to_xarray(var=H, dims=['scenario', 'time', 'node'], coords=[range(d), time_coords, network.n.index.to_list()])
#     last_tp_indeces = [interval_to_var[t] for t in tuplize(tp)]
#     H_sol = H_sol.sel(time=last_tp_indeces)
#     H_sol = H_sol.assign_coords(time=range(len(tp)))
#     # Fill the array with values from the Gurobi solution
#     # for (scenario, time, node), var in H.items():
#     #     H_values[scenario, time, node] = var.X


#     # # Create an xarray DataArray from the NumPy array
#     # H_xarray = xr.DataArray(
#     #     H_values,
#     #     dims=['scenario', 'time', 'node'],
#     #     coords={'scenario': range(d), 'time': range(inst), 'node': range(Nnodes)}
#     # )


#     # #create a list of the variables
#     # variables = ['nh','mhte','meth']

#     #%%if outputs is not empty, create the subplots
#     if outputs != []:
#         fig = make_subplots(rows=len(variables), cols=1)
#         for i, var in enumerate(variables):
#             for j, output in enumerate(outputs):
#                 fig.add_trace(go.Scatter(x=list(range(len(output[0][0]))), y=output[0][i], name=f'Scenario {j+1}'),row=i+1, col=1)
#             fig.update_layout(height=600*len(variables), title=f'{var} variables')
#         fig.show()

#     #otherwise #print a message
#     else:
#         #print("No outputs to plot")

#     #%%

#     fig = px.line(x=["a","b","c"], y=[1,3,2], title="sample figure")
#     #print(fig)
#     fig.show()
#     # %%
