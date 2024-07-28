
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
from YUPPY import OPT1, OPT2, Network, time_partition, df_aggregator
from EU_net import EU
#os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")

# %% class time aggregator

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


def OPT_agg(Network):
    """
	Performs optimization on a given network.
    The network has an initialized time partition to aggregate the data and variables approriately.
    The aggregated model is then solved. The aggregated model is a relaxation of the original model (solved in OPT3).

	Parameters:
	- Network: an instance of the Network class.

	Returns:
	- outputs: a list containing the optimized variables and the objective value.



	"""
    if Network.costs.shape[0] == 1: #if the costs are the same:
        cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly


    start_time=time.time()
    Nnodes = Network.n.shape[0]
    NEedges = Network.edgesP.shape[0]
    NHedges = Network.edgesH.shape[0]
    d = Network.loadP_t_agg.shape[2] #number of scenarios
    inst = Network.loadP_t_agg.shape[0] #number of time steps in time partition
    tp_obj = Network.time_partition
    tp = tp_obj.agg #time partition
    print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
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
    model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge[j,i,k] <= (Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] <= (Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge[j,i,k] >= -(Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge[j,i,k] >= -(Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))

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



    ES = Network.genS_t_agg
    EW = Network.genW_t_agg
    EL = Network.loadP_t_agg
    HL = Network.loadH_t_agg

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
eu.init_time_partition()
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


# print('end iter in {}s'.format(np.round(time.time()-iter_start_time),3))
# #%%
# results_notagg = OPT3(eu)
# %%



"""
Performs optimization on a given network.
given a partition it creates network of the partition. solver the aggregated models, makes the partition finer
then it adds variables and constraints to the previous model to obtain the new aggregated model to have some hope of warmstarting and solves it again.

Parameters:
- Network: an instance of the Network class.

Returns:
- outputs: a list containing the optimized variables and the objective value.



"""
#input parameters
Network = eu


#function code



def add_H_balance_constraint(i0,i1):
    model.addConstrs(- H[j,i1,k] + H[j,i0,k] + 30*Network.n['feth'].iloc[k]*EtH[j,i0,k] - HtE[j,i0,k] -
                                quicksum(H_edge[j,i0,l] for l in Network.edgesH.loc[Network.edgesH['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge[j,i0,l] for l in Network.edgesH.loc[Network.edgesH['end_node']==Network.n.index.to_list()[k]].index.to_list())
                                ==HL[var_to_interval_index[i0],k,j] for j in range(d) for k in range(Nnodes))


if Network.costs.shape[0] == 1: #if the costs are the same:
    cs, cw, ch, chte, ceth, cNTC, cMH = Network.costs['cs'][0], Network.costs['cw'][0], Network.costs['ch'][0], Network.costs['chte'][0], Network.costs['ceth'][0], Network.costs['cNTC'][0], Network.costs['cMH'][0]
else:
    print("add else") #actually we can define the costs appropriately using the network class directly


start_time=time.time()
Nnodes = Network.n.shape[0]
NEedges = Network.edgesP.shape[0]
NHedges = Network.edgesH.shape[0]
d = Network.loadP_t_agg.shape[2] #number of scenarios
inst = Network.loadP_t_agg.shape[0] #number of time steps in time partition
tp_obj = Network.time_partition
tp = tp_obj.agg #time partition
print(f'sanity checl, is inst equal to len tp= {inst == len(tp)}')

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
ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=Network.n['Mns'])
nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=Network.n['Mnw'])
nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=Network.n['Mnh'])
mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=Network.n['Mhte'])
meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=Network.n['Meth'])
addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)

#time and scenario dependent variables
HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
P_edge = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY) #could make sense to sosbstitute Nodes with Network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
#fai due grafi diversi
H_edge = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)

#todo: add starting capacity for generators (the same as for liners)
model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(Nnodes))
model.addConstrs( P_edge[j,i,k] <= (Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
model.addConstrs( H_edge[j,i,k] <= (Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))
model.addConstrs( P_edge[j,i,k] >= -(Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NEedges))
model.addConstrs( H_edge[j,i,k] >= -(Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(i) for i in range(inst) for j in range(d) for k in range(NHedges))


#cons1=model.addConstrs(nh[k]>=0 for k in range(Nnodes) )#for j in range(d) for i in range(inst))
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


ES = Network.genS_t_agg
EW = Network.genW_t_agg
EL = Network.loadP_t_agg
HL = Network.loadH_t_agg

#model.remove(cons1)
for j in range(d):
    for k in range(Nnodes):
        for i in range(inst-1):
            cons2[j,i,k].rhs = HL[i,k,j] #time,node,scenario or if you prefer to not remember use isel
        cons3[j,k].rhs  = HL[inst-1,k,j]


cons1=model.addConstrs(ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*Network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                    quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                    quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['end_node']==Network.n.index.to_list()[k]].index.to_list())
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
for partition_generator in range(1,2):
    #iter over partition_generation
    Network.iter_partition()


    # how to append variables len({**H , **H[list(H.keys())[-2]]})
    # set(a).issubset(b) to check subset relation in lists can be made faster in the class though
    family_tree = Network.time_partition.family_tree

    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

    splitted_intervals = order_intervals(family_tree[-1])
    tp_obj = Network.time_partition #new time partition object
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
        P_edge = {**P_edge, **model.addVars(product(range(d),new_vars,range(NEedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)} #could make sense to sosbstitute Nodes with Network.nodes and so on Nedges with n.edgesP['start_node'],n.edgesP['end_node'] or similar
        H_edge = {**H_edge, **model.addVars(product(range(d),new_vars,range(NHedges)),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)}

        #constraint binding the right interval of the newly split interval to the previous time step
        if is_not_first_round:
            outer_right = interval_to_var[tuple_index]
            add_H_balance_constraint(inner_right,outer_right)

        #modulo constriants
        model.addConstrs( H[j,i,k] <= nh[k] for i in new_vars for j in range(d) for k in range(Nnodes))
        model.addConstrs( EtH[j,i,k] <= meth[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes))
        model.addConstrs( HtE[j,i,k] <= mhte[k]*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(Nnodes))
        model.addConstrs( P_edge[j,i,k] <= (Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges))
        model.addConstrs( H_edge[j,i,k] <= (Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges))
        model.addConstrs( P_edge[j,i,k] >= -(Network.edgesP['NTC'].iloc[k] + addNTC[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NEedges))
        model.addConstrs( H_edge[j,i,k] >= -(Network.edgesH['MH'].iloc[k] + addMH[k])*tp_obj.len(var_to_interval_index[i]) for i in new_vars for j in range(d) for k in range(NHedges))

        #relation between new variables and old variables (i am actually unsure whethere these are necessary)
        model.addConstrs(EtH[j, father_index,k] == quicksum(EtH[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes))
        model.addConstrs(HtE[j, father_index,k] == quicksum(HtE[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(Nnodes))
        model.addConstrs(P_edge[j, father_index,k] == quicksum(P_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NEedges))
        model.addConstrs(H_edge[j, father_index,k] == quicksum(H_edge[j,interval_index_to_var[i],k] for i in split_indeces) for j in range(d) for k in range(NHedges))
        model.addConstrs(H[j,father_index, k] == H[j, interval_index_to_var[split_indeces[-1]],k] for j in range(d) for k in range(Nnodes))
        # # new aggregated scenarios
        ES = Network.genS_t_agg
        EW = Network.genW_t_agg
        EL = Network.loadP_t_agg
        HL = Network.loadH_t_agg

        #Power balance constraints
        model.addConstrs(ns[k]*ES[var_to_interval_index[i],k,j] + nw[k]*EW[var_to_interval_index[i],k,j] + 0.033*Network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                        quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['start_node']==Network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(P_edge[j,i,l] for l in Network.edgesP.loc[Network.edgesP['end_node']==Network.n.index.to_list()[k]].index.to_list())
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
    if last_interval[-1] == Network.T:
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
        


    start_time = time.time()
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
    else:
        VARS+=[[np.ceil([ns[k].X for k in range(Nnodes)]),np.ceil([nw[k].X for k in range(Nnodes)]),np.array([nh[k].X for k in range(Nnodes)]),np.array([mhte[k].X for k in range(Nnodes)]),np.array([meth[k].X for k in range(Nnodes)])]]
        outputs=outputs + [VARS+[model.ObjVal]]
        print("opt time: {}s.".format(np.round(time.time()-start_time,3)))
#%%

# %%
