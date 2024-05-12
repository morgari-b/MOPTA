#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:30:13 2024

@author: morgari
"""

# %% First Cell
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env
import pypsa
from itertools import product

# %%

# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

ES=pd.read_csv('scenarios/PV_scenario100',index_col=0).T
EW=pd.read_csv('scenarios/wind_scenarios.csv',index_col=0).T


#%%
def OPT2(ES,EW,EL,HL,
        #solar el production
        #wind el production
        #electricity load
        #hydrogen load
        #number of scenarios
        cs=4000,cw=3000000,mw=50,ch=7,
        #cost of solar panels
        #cost of wind turbines
        #max socially acceptable wind turbines
        #cost of hydrogen storage
        chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000
        #cost of H to el
        #efficiency of H to el
        #max H to el at an instance
        #cost of el to H
        #efficiency of el to H
        #max el to H at an instance
        ):
        #option to insert the graph
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    #from graph gather this info:
    d=1
    inst=365*24
    Ns = 1
    Nw = 1
    Nel = 7 #nodes of electricity load
    Nhl = 2
    Nz = 1
    Nfc = 1
    czfc = 13#*np.ones([Nz,Nfc]) #random
    czhl = 0#*np.ones(Nz,Nhl) #cost matrix Nz x Nhl of cost of transport
    
    # variables linked to the NODES:
        
    # max PV and turbines per farm
    ns = model.addVars(Ns,vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVars(Nw,vtype=GRB.INTEGER, obj=cw,lb=0)    
    # needed max capacity at electrolyzers z, fuel cells fc
    nhz = model.addVars(Nz,vtype=GRB.CONTINUOUS, obj=ch,lb=0)
    nhfc = model.addVars(Nfc,vtype=GRB.CONTINUOUS, obj=ch,lb=0)
    # at each time step: conversion to and from H2 at each z and fc, H2 storage, max needed conversion per hour:
    HtE = model.addVars(product(range(Nfc),range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(Nz),range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in MWh
    Hz = model.addVars(product(range(Nz),range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
    Hfc = model.addVars(product(range(Nfc),range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
    mhte=model.addVars(Nfc,vtype=GRB.CONTINUOUS,obj=0.01)
    meth=model.addVars(Nz,vtype=GRB.CONTINUOUS,obj=0.01)   # POTENTIAL CHANGE: ask policymaker how much cost to improve conversion per hour??
    
    # variables linked to the EDGES:
    
    # electric grid:
    s_el = model.addVars(product(range(Ns),range(Nel),range(d),range(inst)))
    s_z = model.addVars(product(range(Ns),range(Nz),range(d),range(inst)))
    w_el = model.addVars(product(range(Nw),range(Nel),range(d),range(inst)))
    w_z = model.addVars(product(range(Nw),range(Nz),range(d),range(inst)))
    fc_el = model.addVars(product(range(Nfc),range(Nel),range(d),range(inst)))
    # hydrogen road transport:
    z_fc = model.addVars(product(range(Nz),range(Nfc),range(d),range(365)), obj=czfc) # ASSUMPTION: trucks depart at 8 in the morning, once a day, with predicted H2 industrial load / needed fuel cell storage for the whole day.(32%24==8)*var
    z_hl = model.addVars(product(range(Nz),range(Nhl),range(d),range(365)), obj=czhl)
    # POSSIBLE ADDITION: connect generator nodes with each other? To avoid congestion?
    
    
    # constraints
    for j in range(d):
        
        # hydrogen at hl:
        for i in range(365):
            model.addConstrs(quicksum(HL[0,i+ii] for ii in range(24))==quicksum(z_hl[z,hl,j,i] for z in range(Nz)) for hl in range(Nhl)) #assume you get at 8 all load for next day, HL[hl,j,i]
               
        for i in range(inst-1):
            # electricity at s,w,z,fc,el nodes:
            model.addConstrs(quicksum(s_el[s,el,j,i] for el in range(Nel)) + 
                             quicksum(s_z[s,z,j,i] for z in range(Nz)) <= 
                             ns[s]*ES[j,i] for s in range(Ns))                       # electricity exiting at every PV farm, ES[s,j,i] format?????
            model.addConstrs(quicksum(w_el[w,el,j,i] for el in range(Nel)) + 
                             quicksum(w_z[w,z,j,i] for z in range(Nz)) <= 
                             nw[w]*EW[j,i] for w in range(Nw))                       # electricity exiting at every wind farm
            model.addConstrs(quicksum(s_z[s,z,j,i] for s in range(Ns)) +
                             quicksum(w_z[w,z,j,i] for w in range(Nw)) ==
                             EtH[z,j,i] for z in range(Nz))                    # electricity converted to H2 at every elecrolyzer
            model.addConstrs(quicksum(fc_el[fc,el,j,i] for el in range(Nel)) == 
                             0.044*fhte*HtE[fc,j,i] for fc in range(Nfc))      # electricity converted from H2 at every fuel cell: assumes it all goes to load, no other grid connections
            model.addConstrs(quicksum(s_el[s,el,j,i] for s in range(Ns)) + 
                             quicksum(w_el[w,el,j,i] for w in range(Nw)) + 
                             quicksum(fc_el[fc,el,j,i] for fc in range(Nfc)) ==
                             EL[0,i] for el in range(Nel))                     # electricity consumed at every load node, EL[el,j,i]
            
            # hydrogen at z,fc:
            model.addConstrs(Hz[z,j,i+1]==Hz[z,j,i]+28.5*feth*EtH[z,j,i] -     # comment on conversion
                             quicksum(z_hl[z,hl,j,i//24] for hl in range(Nhl))*(i%24==8) -
                             quicksum(z_fc[z,fc,j,i//24] for fc in range(Nfc))*(i%24==8) for z in range(Nz))
            model.addConstrs(Hfc[fc,j,i+1]==Hfc[fc,j,i]-HtE[fc,j,i] +
                             quicksum(z_fc[z,fc,j,i//24]*(i%24==12) for z in range(Nz)) for fc in range(Nfc))
            model.addConstrs(Hz[z,j,i]<=nhz[z] for z in range(Nz))
            model.addConstrs(Hfc[fc,j,i]<=nhfc[fc] for fc in range(Nfc))
            
            # convergence efficiency at z and fc:
            model.addConstrs(EtH[z,j,i]<=meth[z] for z in range(Nz))
            model.addConstrs(HtE[fc,j,i]<=mhte[fc] for fc in range(Nfc))
    model.addConstrs( meth[z] <= Meth for z in range(Nz))
    model.addConstrs( mhte[fc] <= Mhte for fc in range(Nfc))   
    
    #maximum wind turbines at a farm:
    model.addConstrs(nw[w]<=mw for w in range(Nw))
    
    model.optimize()
    #if model.Status!=2:
    print("Status = {}".format(model.Status))
    return None



#%%


#def OPT2(ES,EW,EL,HL,
        #solar el production
        #wind el production
        #electricity load
        #hydrogen load
        #number of scenarios
cs=4000
cw=3000000
mw=50
ch=7
        #cost of solar panels
        #cost of wind turbines
        #max socially acceptable wind turbines
        #cost of hydrogen storage
chte=0
fhte=0.75
Mhte=200000
ceth=0
feth=0.7
Meth=15000
        #cost of H to el
        #efficiency of H to el
        #max H to el at an instance
        #cost of el to H
        #efficiency of el to H
        #max el to H at an instance
   #     ):
        #option to insert the graph

env = Env(params={'OutputFlag': 0})
model = Model(env=env)
inst=24*365

#from graph gather this info:
d=1
Ns = 1
Nw = 1
Nel = 7 #nodes of electricity load
Nhl = 2
Nz = 1
Nfc = 1
czfc = 13#*np.ones([Nz,Nfc]) #random
czhl = 0#*np.ones(Nz,Nhl) #cost matrix Nz x Nhl of cost of transport

# variables linked to the NODES:
    
# max PV and turbines per farm
ns = model.addVars(Ns,vtype=GRB.INTEGER, obj=cs,lb=0)
nw = model.addVars(Nw,vtype=GRB.INTEGER, obj=cw,lb=0)    
# needed max capacity at electrolyzers z, fuel cells fc
nhz = model.addVars(Nz,vtype=GRB.CONTINUOUS, obj=ch,lb=0)
nhfc = model.addVars(Nfc,vtype=GRB.CONTINUOUS, obj=ch,lb=0)
# at each time step: conversion to and from H2 at each z and fc, H2 storage, max needed conversion per hour:
HtE = model.addVars(product(range(Nfc),range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte,lb=0) # expressed in kg      
EtH = model.addVars(product(range(Nz),range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in MWh
Hz = model.addVars(product(range(Nz),range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
Hfc = model.addVars(product(range(Nfc),range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
mhte=model.addVars(Nfc,vtype=GRB.CONTINUOUS,obj=0.01)
meth=model.addVars(Nz,vtype=GRB.CONTINUOUS,obj=0.01)   # POTENTIAL CHANGE: ask policymaker how much cost to improve conversion per hour??

# variables linked to the EDGES:

# electric grid:
s_el = model.addVars(product(range(Ns),range(Nel),range(d),range(inst)))
s_z = model.addVars(product(range(Ns),range(Nz),range(d),range(inst)))
w_el = model.addVars(product(range(Nw),range(Nel),range(d),range(inst)))
w_z = model.addVars(product(range(Nw),range(Nz),range(d),range(inst)))
z_el = model.addVars(product(range(Nz),range(Nel),range(d),range(inst)))
# hydrogen road transport:
z_fc = model.addVars(product(range(Nz),range(Nfc),range(d),range(365)), obj=czfc) # ASSUMPTION: trucks depart at 8 in the morning, once a day, with predicted H2 industrial load / needed fuel cell storage for the whole day.(32%24==8)*var
z_hl = model.addVars(product(range(Nz),range(Nhl),range(d),range(365)), obj=czhl)
# POSSIBLE ADDITION: connect generator nodes with each other? To avoid congestion?


# constraints
for j in range(d):
    for i in range(inst):
        # electricity at s,w,z,el nodes:
        #model.addConstrs([quicksum(s_el[s,el,j,i] for el in range(Nel)) <= ES[i] for s in range(Ns)]) #ES[s,j,i] format?????
        model.addConstrs(Hz[z,j,i] <= 0 for z in range(Nz))
        #model.addConstr( quicksum(EL[k,j,i] for k in range(Nel)) + quicksum(EtH[k,j,i] for k in range(Nz) <= quicksum(0.044*fhte*HtE[k,j,i] for k in range(fc) + quicksum(ns[k]*ES[k,j,i] for k in range(Ns))) + quicksum(nw[k]*EW[k,j,i] for k in range(Nw))))
        # hydrogen at z,fc,hl nodes:
        #model.addConstr( H[j,i] == H[j,i-1] + 28.5*feth*EtH[j,i] - HL[j,i] - HtE[j,i] )
        #model.addConstr( H[j,i] <= nh )
        #model.addConstr( EtH[j,i] <= meth)
        #model.addConstr( HtE[j,i] <= mhte)
#model.addConstr( nw <= mw )
#model.addConstr( meth <= Meth )
#model.addConstr( mhte <= Mhte )     


#model.optimize()
#if model.Status!=2:
#print("Status = {}".format(model.Status))
#return None


