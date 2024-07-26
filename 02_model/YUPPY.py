#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:05:49 2024

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

#%% import_generated_scenario

#LESS IMPORTANT:
# give name to constriants
#todo: da spostrare in uno script più sensato
def import_generated_scenario(path, nrows,  scenario, node_names = None):
    """
    Import a generated scenario from a CSV file.
    Args:            
        path (str): The path to the CSV file.
        nrows (int): The number of rows to read from the CSV file (generally the number of nodes).
        scenario (str): The name of the scenario.
        nodes (list): list of str containing the name of the noddes
    Returns:
        xr.DataArray: The imported scenario as a DataArray with dimensions 'time', 'node', and 'scenario'.
            The 'time' dimension represents the time index, the 'node' dimension represents the nodes, and the 'scenario'
            dimension represents the scenario name.

   """
    df = pd.read_csv(path, index_col = 0).head(nrows)
    time_index = range(df.shape[1])#pd.date_range('2023-01-01 00:00:00', periods=df.shape[1], freq='H')
    if node_names == None:
        node_names = df.index

    scenario = xr.DataArray(
        np.expand_dims(df.T.values, axis = 2),
        coords={'time': time_index, 'node': node_names, 'scenario': [scenario]},
        dims=['time', 'node', 'scenario']
    )
    return scenario

#%% OPT1 - single node
def OPT1(es,ew,el,hl,d=5,rounds=4,cs=4000, cw=3000000,ch=10,Mns=10**5,Mnw=500,Mnh=10**9,chte=2,fhte=0.75,Mhte=10**6,ceth=200,feth=0.7,Meth=10**5):
            
    start_time=time.time()
    
    D,inst = np.shape(es)
    rounds=min(rounds,D//d)
    print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    
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
    
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
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
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE


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
    
    def __init__(self):
        """
        Initialize a Network object.
        
        Parameters:
            nodes (pandas.DataFrame): DataFrame with index name of node, columns lat and long.
            edgesP (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
            edgesH (pandas.DataFrame): DataFrame with columns start node and end node, start coords and end coords.
        """
        self.n=pd.DataFrame(columns=['node','lat','long']).set_index('node')
        # nodes = pd.DataFrame with index name of node, columns lat and long.
        self.edgesP=pd.DataFrame(columns=['start_node','end_node'])
        self.edgesH=pd.DataFrame(columns=['start_node','end_node'])
        # edges = pd.DataFrame with columns start node and end node, start coords and end coords.
       
        self.costs = pd.DataFrame(columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])
  
    
    def plot(self):
        """
        Plot the network using folium.
        """
        loc = [self.n['lat'].mean(), self.n["long"].mean()]
        m = folium.Map(location=loc, zoom_start=5, tiles='CartoDB positron')
        
        for node in self.n.index.to_list():
            folium.Marker(
                location=(self.n.loc[node, 'lat'],self.n.loc[node, 'long']),
                icon=folium.Icon(color="green"),
            ).add_to(m)
        for edge in self.edgesP.index.to_list():
            start_node, end_node = self.edgesP.loc[edge,'start_node'],self.edgesP.loc[edge,'end_node']
            start_loc, end_loc = (self.n.loc[start_node, 'lat'],self.n.loc[start_node, 'long']),(self.n.loc[end_node, 'lat'],self.n.loc[end_node, 'long'])
            folium.PolyLine([start_loc,end_loc],weight=5,color='blue', opacity=.2).add_to(m)
        
        #A=[[[elem[1],elem[0]] for elem in list(x.exterior.coords)] for x in list(data_hex2['geometry'])]
        #for hex in A:
         # folium.PolyLine(hex,weight=5,color='blue', opacity=.2).add_to(m)
        m.save("Network.html")

        
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



    
#%% OPT2
def OPT2(Network, d=1,rounds=1,long_outs=False):
    
    if Network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    

    start_time=time.time()
    Nnodes = Network.n.shape[0]
    NEedges = Network.edgesP.shape[0]
    NHedges = Network.edgesH.shape[0]
    D = Network.loadP_t.shape[2] #number of scenarios
    inst = Network.loadP_t.shape[0] #number of time steps T
    rounds=min(rounds,D//d)
    print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
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
    HX=Network.genW_t.copy()
    HX[:,:,:]=0
    EtHX=HX.copy()
    HtEX=HX.copy()
    P_edgeX=HX.copy()
    H_edgeX=HX.copy()
    
    #todo perchè lo rimuoviamo questo? # perché è quello che va sostituito in toto ad ogni giro... è qui come fermaposto
    cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
    #todo: sobstitute 30 with a parameter # NO!! 30 e poi sotto 0.033 sono i rate di conversione kg di idrogeno - MW se la conversione fosse 100% efficiente, non sono parametri.
    cons2=model.addConstrs(- H[j,i+1,k] + H[j,i,k] + 30*Network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -  
                           quicksum(H_edge[j,i,l] for l in Network.edgesH.loc[Network.edgesH['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,i,l] for l in Network.edgesH.loc[Network.edgesH['end_node']==Network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for i in range(inst-1) for k in range(Nnodes))
    cons3=model.addConstrs(- H[j,0,k] + H[j,inst-1,k] + 30*Network.n['feth'].iloc[k]*EtH[j,inst-1,k] - HtE[j,inst-1,k] -
                           quicksum(H_edge[j,inst-1,l] for l in Network.edgesH.loc[Network.edgesH['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                           quicksum(H_edge[j,inst-1,l] for l in Network.edgesH.loc[Network.edgesH['end_node']==Network.n.index.to_list()[k]].index.to_list())
                           ==0 for j in range(d) for k in range(Nnodes))
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        gr_start_time=time.time()

        ES = Network.genS_t.sel(scenario = slice(d*group, d*(group+1)))
        EW = Network.genW_t.sel(scenario = slice(d*group, d*(group+1)))
        EL = Network.loadP_t.sel(scenario = slice(d*group, d*(group+1)))
        HL = Network.loadH_t.sel(scenario = slice(d*group, d*(group+1)))

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


# %% OPT3

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
