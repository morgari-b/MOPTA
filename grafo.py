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

EL=pd.read_excel('data.xlsx',sheet_name='Electricity Load')
GL=pd.read_excel('data.xlsx',sheet_name='Gas Load')
S=pd.read_excel('data.xlsx',sheet_name='Solar')
W=pd.read_excel('data.xlsx',sheet_name='Wind')

el=[]
gl=[]
s=[]
w=[]
for i in range(4):
    el = el + 4*EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby((EL["Instance"]-1)//4)["Load"].sum().to_list()
    gl = gl + 4*GL.loc[GL["Quarter"]=="Q{}".format(i+1)].groupby((GL["Instance"]-1)//4)["Load"].sum().to_list()
    s = s + 4*S.loc[S["Quarter"]=="Q{}".format(i+1)].groupby((S["Instance"]-1)//4)["Generation"].sum().to_list()
    w = w + 4*W.loc[W["Quarter"]=="Q{}".format(i+1)].groupby((W["Instance"]-1)//4)["Generation"].sum().to_list()
el = np.matrix(el)
gl = np.matrix(gl)
s = np.matrix(s)
w = np.matrix(w)

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
    inst=384
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
        for i in range(inst):
            # electricity at s,w,z,fc,el nodes:
            model.addConstrs(quicksum(s_el[s,el,j,i] for el in range(Nel)) + 
                             quicksum(s_z[s,z,j,i] for z in range(Nz)) <= 
                             ES[0,i] for s in range(Ns))                       # electricity exiting at every PV farm, ES[s,j,i] format?????
            model.addConstrs(quicksum(w_el[w,el,j,i] for el in range(Nel)) + 
                             quicksum(w_z[w,z,j,i] for z in range(Nz)) <= 
                             ES[0,i] for w in range(Nw))                       # electricity exiting at every wind farm
            model.addConstrs(quicksum(s_z[s,z,j,i] for s in range(Ns)) +
                             quicksum(w_z[w,z,j,i] for w in range(Nw)) ==
                             EtH[z,j,i] for z in range(Nz))                    # electricity converted to H2 at every elecrolyzer
            model.addConstrs(quicksum(fc_el[fc,el,j,i] for el in range(Nel)) == 
                             0.044*fhte*HtE[fc,j,i] for fc in range(Nfc))      # electricity converted from H2 at every fuel cell: assumes it all goes to load, no other grid connections
            model.addConstrs(quicksum(s_el[s,el,j,i] for s in range(Ns)) + 
                             quicksum(w_el[w,el,j,i] for w in range(Nw)) + 
                             quicksum(fc_el[fc,el,j,i] for fc in range(Nfc)) ==
                             EL[0,i] for el in range(Nel))                     # electricity consumed at every load node, EL[el,j,i]
            
            # hydrogen at z,fc,hl:
            model.addConstrs(quicksum(quicksum(z_hl[z,hl,j,i+ii] for hl in range(Nhl)) + 
                                      quicksum(z_fc[z,fc,j,i+ii] for fc in range (Nfc)) for ii in range(24)) ==
                             28.5*feth*EtH[z,j,i] for z in range(Nz)) 
            
            # NO STO SBAGLIANDO TUTTO : somme sulle 24h sono al load, mi dicono quanto togliere alle 8
            
            #model.addConstrs(Hz[z,j,i] <= 0 for z in range(Nz))
            #model.addConstr( quicksum(EL[k,j,i] for k in range(Nel)) + quicksum(EtH[k,j,i] for k in range(Nz) <= quicksum(0.044*fhte*HtE[k,j,i] for k in range(fc) + quicksum(ns[k]*ES[k,j,i] for k in range(Ns))) + quicksum(nw[k]*EW[k,j,i] for k in range(Nw))))
            # hydrogen at z,fc,hl nodes:
            #model.addConstr( H[j,i] == H[j,i-1] + 28.5*feth*EtH[j,i] - HL[j,i] - HtE[j,i] )
            #model.addConstr( H[j,i] <= nh )
            #model.addConstr( EtH[j,i] <= meth)
            #model.addConstr( HtE[j,i] <= mhte)
    #model.addConstr( nw <= mw )
    #model.addConstr( meth <= Meth )
    #model.addConstr( mhte <= Mhte )     
    
    model.optimize()
    #if model.Status!=2:
    print("Status = {}".format(model.Status))
    return None



#%%

ES=s
EW=w
EL=el
HL=gl

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


