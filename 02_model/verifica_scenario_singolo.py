#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:17:17 2024

@author: morgari


VERIFICA CHE IL VECCHIO MODELLO SU NODO SINGOLO E IL NUOVO MODELLO A PIÙ NODI APPLICATO AD UN NODO SINGOLO COINCIDANO. RISPOSTA: SÌ COINCIDONO.


"""

#%% import modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates

import xarray as xr
import folium

from YUPPY import OPT1, OPT2, Network


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

#%% OPT2 - network
def OPT2(Network,
         d=1,rounds=1
         ):
    
    if Network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    

    start_time=time.time()
    Nnodes = Network.n.shape[0]
    NEedges = Network.edgesP.shape[0]
    NHedges = Network.edgesH.shape[0]
    D = Network.loadE_t.shape[2] #number of scenarios
    inst = Network.loadE_t.shape[0] #number of time steps T
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
    
    for group in range(rounds):
        gr_start_time=time.time()

        # PER GABOR!!!!
        # così sto prendendo gli stati non gli scenari vanno messi i nodi.
        # idea: cambia struttura di Network.loadE etc dipendenti dal tempo in modo da avere i paesi come colonna e un'altra colonna con "scenario"
        # così da avere semplicemente più righe con gli scenari per ogni paese.
        # aggiusta qui usando lo stesso trucchetto di sotto con ==.
        ES = Network.genS_t.sel(scenario = slice(d*group, d*(group+1)))
        EW = Network.genW_t.sel(scenario = slice(d*group, d*(group+1)))
        EL = Network.loadE_t.sel(scenario = slice(d*group, d*(group+1)))
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
            
    return outputs#,HH,ETH,HTE


#%% prepare data for both methods

# data for OPT1

ES=0.015*pd.read_csv('MOPTA/01_scenario_generation/scenarios/PV_scenario100.csv',index_col=0).head(1)
EW=4*pd.read_csv('MOPTA/01_scenario_generation/scenarios/wind_scenarios.csv',index_col=0).head(1)
EL=pd.DataFrame(450*pd.read_csv('MOPTA/01_scenario_generation/scenarios/electricity_load_2023.csv',index_col='DateUTC')['BE']).T
HL=pd.read_csv('MOPTA/01_scenario_generation/scenarios/hydrogen_demandg.csv',index_col=0).head(1)

# data for OPT2

EU=pd.DataFrame([['Italy',41.90,12.48]], columns=['node','lat','long']).set_index('node')
EU_e=pd.DataFrame(columns=['start_node','end_node']) #empty because only one node
EU_h=pd.DataFrame(columns=['start_node','end_node'])
EU['Mhte']=10**6  # maximum hydrogen transport cost
EU['Meth']=10**5  # maximum electricity transport cost
EU['feth']=0.7  # fraction of electricity in hydrogen
EU['fhte']=0.75  # fraction of electricity in hydrogen
EU['Mns'] = 100000
EU['Mnw'] = 10000
EU['Mnh'] = 10000000
EU_e['NTC']=1000  # maximum transportation cost for electricity
EU_h['MH']=500  # maximum transportation cost for hydrogen
costs = pd.DataFrame([["Italy",4000, 3000000, 10,2,200,1000,10000]],columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])

eu = Network()
eu.n = EU
eu.edgesP = EU_e
eu.edgesH = EU_h
eu.costs = costs
eu.genW_t = xr.DataArray(np.expand_dims(EW.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=EW.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.genS_t = xr.DataArray(np.expand_dims(ES.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=ES.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.loadH_t = xr.DataArray(np.expand_dims(HL.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=HL.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.loadE_t = xr.DataArray(np.expand_dims(EL.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=EL.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )


#%% compute results

out_sing1 = OPT1(ES.to_numpy(),EW.to_numpy(),EL.to_numpy(),HL.to_numpy(),d=1,rounds=1)
out_sing2 = OPT2(eu)


#%% results

"""

STARTING OPT1 -- setting up model for 1 batches of 1 scenarios.

OPT Model has been set up, this took  0.3647 s.
Round 1 of 1 - opt time: 1.035s.



STARTING OPT2 -- setting up model for 1 batches of 1 scenarios.

OPT Model has been set up, this took  2.4534 s.
Round 1 of 1 - opt time: 14.222s.



out_sing1
Out[118]: 
[[87328.0,
  451.0,
  3333390.141516082,
  14053.347524572586,
  1027.2159299583018,
  2082491174.8506062]]

out_sing2
Out[117]: 
[[array([87328.]),
  array([451.]),
  array([3333391.]),
  array([14054.]),
  array([1028.]),
  2082491174.8506062]]


Conclusion: on the same scenario, single node, the two methods give the same result, but the second takes much longer to compute.

"""


