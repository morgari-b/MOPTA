# %% Import packages

import numpy as np
import pandas as pd
import xarray as xr
import folium
from gurobipy import Model, GRB, Env, quicksum
import gurobipy as gp
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #, mdates

#IMPORTANT TODOS:

#LESS IMPORTANT TODOS:
# -ogni tanto usiamo h e altre volte H per l'idrogeno
#%%
def clearvars():    
    for el in sorted(globals()):
        if '__' not in el:
                print(f'deleted: {el}')
                del el

#%%
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
    time_index = pd.date_range('2023-01-01 00:00:00', periods=df.shape[1], freq='H')
    if node_names == None:
        node_names = df.index

    scenario = xr.DataArray(
        np.expand_dims(df.T.values, axis = 2),
        coords={'time': time_index, 'node': node_names, 'scenario': [scenario]},
        dims=['time', 'node', 'scenario']
    )
    return scenario

#%%
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

    

#%%

EU=pd.DataFrame([['Italy',41.90,12.48],
                ['Spain',40.42,-3.70],
                ['Austria',48.21,16.37],
                ['France',48.86,2.35]], 
                columns=['node','lat','long']).set_index('node')
             
#ho spostato qui così è un parametro di input
EU['Mhte']=10**6  # maximum hydrogen transport cost
EU['Meth']=10**5  # maximum electricity transport cost
EU['feth']=0.7  # fraction of electricity in hydrogen
EU['fhte']=0.75  # fraction of electricity in hydrogen
EU['Mns'] = 10000
EU['Mnw'] = 10000
EU['Mnh'] = 10000

EU_e=pd.DataFrame([  ['France','Italy'],
                     ['Austria','Italy'],
                     ['France','Spain'],
                     ['France','Austria']  ],columns=['start_node','end_node'])

EU_h=pd.DataFrame([  ['France','Italy'],
                     ['Austria','Italy'],
                     ['France','Spain'],
                     ['France','Austria']  ],columns=['start_node','end_node'])

#ho spostato qui così è un parametro di input
EU_e['NTC']=1000  # maximum transportation cost for electricity
EU_h['MH']=500  # maximum transportation cost for hydrogen

#costs
costs = pd.DataFrame([["All",4000, 3000000, 10,2,200,1000,10000]],columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])

eu = Network()

eu.n = EU
eu.edgesP = EU_e
eu.edgesH = EU_h
eu.costs = costs

# eu.loadE[1]=pd.read_csv('scenarios/electricity_load.csv',usecols=['IT','ES','AT','FR']).set_index(pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')).T.set_index(eu.n.index)
# eu.genW[1]=pd.read_csv('scenarios/wind_scenarios.csv',index_col=0).head(4).set_index(eu.n.index)
# eu.genS[1]=pd.read_csv('scenarios/PV_scenario.csv',index_col=0).head(4).set_index(eu.n.index)
# eu.loadH[1]=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).head(4).set_index(eu.n.index)

#%%

# Import scenarios
elec_load_df = pd.read_csv('scenarios/electricity_load.csv')
elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR']]
time_index = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

elec_load_scenario = xr.DataArray(
    np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR']].values, axis = 2), #add one dimension to correspond with scenarios
    coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France'], 'scenario': [0]},
    dims=['time', 'node', 'scenario']
)

scenario = 0
wind_scenario = import_generated_scenario('scenarios/wind_scenarios.csv',4,scenario, node_names= ['Italy', 'Spain', 'Austria', 'France'])
pv_scenario = import_generated_scenario('scenarios/PV_scenario.csv',4, scenario, node_names=['Italy', 'Spain', 'Austria', 'France'])
hydrogen_demand_scenario = import_generated_scenario('scenarios/hydrogen_demandg.csv',4, scenario, node_names=['Italy', 'Spain', 'Austria', 'France'])

eu.genW_t = wind_scenario
eu.genS_t = pv_scenario
eu.loadH_t = hydrogen_demand_scenario
eu.loadE_t = elec_load_scenario

#%% to do: make time frames smaller
#eu.genW_t.isel(scenario = [0],node = slice(1,3)).isel(node = 0)
eu.genW_t.shape

#%%
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
    D = eu.loadE_t.shape[2] #number of scenarios
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
    cons2=model.addConstrs(  H[j,i,k] - H[j,i+1,k] + 30*Network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -  
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

    
        ES = Network.genS_t.sel(scenario = slice(d*group, (d+1)*group))
        EW = Network.genW_t.sel(scenario = slice(d*group, (d+1)*group))
        EL = Network.loadE_t.sel(scenario = slice(d*group, (d+1)*group))
        HL = Network.loadH_t.sel(scenario = slice(d*group, (d+1)*group))

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
            VARS=[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.ceil([nh[k].X for k in range(Nnodes)]),np.ceil([mhte[k].X for k in range(Nnodes)]),np.ceil([meth[k].X for k in range(Nnodes)])]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE
# %%
eu.costs.shape[0]
outputs = OPT2(eu)

#%% Ho riscritto il modello in maniera un po' più canon
def OPT3(Network,
         d=1,rounds=1
         ):
    
    if Network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    

    start_time=time.time()
    
    Nnodes = Network.n.shape[0]
    NPedges = Network.edgesP.shape[0]
    NHedges = Network.edgesH.shape[0]
    D = eu.loadE_t.shape[2] #total number of scenarios
    inst = Network.loadE_t.shape[0] #number of time steps T
    rounds=min(rounds,D//d)

    nodes = Network.n.index.to_list()
    scenarios = range(d)
    time_steps = range(inst)
    Pedges = list(zip(eu.edgesP['start_node'].to_list(),eu.edgesP['end_node'].to_list()))
    #print(Pedges.keys())
    Hedges = list(zip(eu.edgesH['start_node'].to_list(),eu.edgesH['end_node'].to_list()))
    print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    
    ns = model.addVars(nodes,vtype=GRB.CONTINUOUS, obj=cs,ub=Network.n['Mns'])
    nw = model.addVars(nodes,vtype=GRB.CONTINUOUS, obj=cw,ub=Network.n['Mnw'])    
    nh = model.addVars(nodes,vtype=GRB.CONTINUOUS, obj=ch,ub=Network.n['Mnh'])   
    mhte = model.addVars(nodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=Network.n['Mhte'])
    meth = model.addVars(nodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=Network.n['Meth'])
    addNTC = model.addVars(Pedges,vtype=GRB.CONTINUOUS,obj=cNTC) 
    addMH = model.addVars(Hedges,vtype=GRB.CONTINUOUS,obj=cMH) 
    
    HtE = model.addVars(product(scenarios,time_steps,nodes),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(scenarios,time_steps,nodes),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(scenarios,time_steps,nodes),vtype=GRB.CONTINUOUS,lb=0)
    P_edge = model.addVars(product(scenarios,time_steps,Pedges),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with Network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
    print(P_edge.keys())
    #fai due grafi diversi
    H_edge = model.addVars(product(scenarios,time_steps,Hedges),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,t,k] <= nh[k] for t in time_steps for j in scenarios for k in nodes) 
    model.addConstrs( EtH[j,t,k] <= meth[k] for t in time_steps for j in scenarios for k in nodes)
    model.addConstrs( HtE[j,t,k] <= mhte[k] for t in time_steps for j in scenarios for k in nodes)
    model.addConstrs( P_edge[j,t,(n0,n1)] <= Network.edgesP['NTC'] + addNTC[(n0,n1)] for t in time_steps for j in scenarios for (n0,n1) in Pedges)
    model.addConstrs( H_edge[j,t,(n0,n1)] <= Network.edgesH['MH'].iloc[k] + addMH[k] for t in time_steps for j in scenarios for (n0,n1) in Hedges)

    outputs=[]
    VARS=[]
    #todo perchè lo rimuoviamo questo?
    cons1=model.addConstrs(nh[n]>=0 for n in nodes )#for j in range(d) for i in time_steps)
    #todo: sobstitute 30 with a parameter
    cons2=model.addConstrs(  H[j,t,k] - H[j,t+1,n] + 30*Network.n['feth'].iloc[k]*EtH[j,t,k] - HtE[j,t,k] -  
                           quicksum(H_edge[j,t,(n,m)] for m in nodes if (m,n) in Hedges) +
                           quicksum(H_edge[j,t,(m,n)] for m in nodes if (n,m) in Hedges)
                           ==0 for j in scenarios for t in time_steps[:-1] for n in nodes)
    cons3=model.addConstrs(- H[j,0,n] + H[j,inst-1,n] + 30*Network.n['feth'].iloc[n]*EtH[j,inst-1,n] - HtE[j,inst-1,n] -
                           quicksum(H_edge[j,inst-1,(n,m)] for m in nodes if (n,m) in Hedges) +
                           quicksum(H_edge[j,inst-1,(m,n)] for m in nodes if (m,n) in Hedges)
                           ==0 for j in scenarios for n in nodes)
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        gr_start_time=time.time()

    
        ES = Network.genS_t.sel(scenario = slice(d*group, (d+1)*group))
        EW = Network.genW_t.sel(scenario = slice(d*group, (d+1)*group))
        EL = Network.loadE_t.sel(scenario = slice(d*group, (d+1)*group))
        HL = Network.loadH_t.sel(scenario = slice(d*group, (d+1)*group))

        model.remove(cons1)
        for j in scenarios: 
            for k in nodes:
                for i in range(inst-1):
                    cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel     
                cons3[j,k].rhs  = HL[inst-1,k,j]
        
        try:    
            cons1=model.addConstrs(ns[n]*ES[t,n,j] + nw[n]*EW[t,n,j] + 0.033*Network.n['fhte'].iloc[n]*HtE[j,t,n] - EtH[j,t,n] -
                                quicksum(P_edge[j,t,(n,m)] for m in nodes if (m,n) in Pedges) +
                                quicksum(P_edge[j,i,(m,n)] for m in nodes if (n,m) in Pedges) 
                                >= EL[i,k,j] for n in nodes for j in scenarios for t in time_steps)
        except IndexError as e:
            print(f"IndexError occurred at t={t}, n={n}, j={j}")
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
            VARS=[np.ceil([ns[k].X for k in nodes]),np.ceil([nw[k].X for k in nodes]),np.ceil([nh[k].X for k in nodes]),np.ceil([mhte[k].X for k in nodes]),np.ceil([meth[k].X for k in nodes])]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE


# %% 
results = OPT3(eu)
# %%
