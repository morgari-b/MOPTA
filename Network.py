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
    coords={'time': time_index, 'node': ['IT', 'ES', 'AT', 'FR'], 'scenario': [1]},
    dims=['time', 'node', 'scenario']
)

scenario = 0
wind_scenario = import_generated_scenario('scenarios/wind_scenarios.csv',4,scenario, node_names= ['IT', 'ES', 'AT', 'FR'])
pv_scenario = import_generated_scenario('scenarios/PV_scenario.csv',4, scenario, node_names= ['IT', 'ES', 'AT', 'FR'])
hydrogen_demand_scenario = import_generated_scenario('scenarios/hydrogen_demandg.csv',4, scenario, node_names= ['IT', 'ES', 'AT', 'FR'])

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
        ES = Network.genS_t.sel(scenario = slice(d*group, (d+1)*group))
        EW = Network.genW_t.sel(scenario = slice(d*group, (d+1)*group))
        EL = Network.loadE_t.sel(scenario = slice(d*group, (d+1)*group))
        HL = Network.loadH_t.sel(scenario = slice(d*group, (d+1)*group))

        model.remove(cons1)
        for j in range(d): 
            for k in range(Nnodes):
                for i in range(inst-1):
                    cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel
                    print(ES[i,k,j])
                cons3[j,k].rhs  = HL[inst-1,k,j]
        cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*Network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                               quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                               quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['end_node']==Network.n.index.to_list()[k]].index.to_list()) 
                               >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[np.ceil(ns.X),np.ceil(nw.X),nh.X,mhte.X,meth.X]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE
# %%
eu.costs.shape[0]
ouputs = OPT2(eu)
# %%

# %%
