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

ES=0.015*pd.read_csv('scenarios/PV_scenario100.csv',index_col=0).to_numpy()
EW=4*pd.read_csv('scenarios/wind_scenarios.csv',index_col=0).to_numpy()
EL=np.zeros([7,100,8760])
for i,loc in enumerate(['1_i','2_i','3_r','4_r','5_r','1_r','2_r']):
    el=pd.read_csv('scenarios/electric_demand{}.csv'.format(loc),index_col=0).to_numpy()
    EL[i,:,:]=el
HL=np.zeros([1,100,8760])
HL[0,:,:]=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()



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
    Nel,d,inst=np.shape(EL)
    Ns = 1
    Nw = 1
    Nhl = 1
    Nz = 1
    Nfc = 1
    czfc = 13#*np.ones([Nz,Nfc]) #random
    czhl = 0#*np.ones(Nz,Nhl) #cost matrix Nz x Nhl of cost of transport
    
    # variables linked to the NODES:
        
    # max PV and turbines per farm
    ns = model.addVars(Ns,vtype=GRB.CONTINUOUS, obj=cs,lb=0)
    nw = model.addVars(Nw,vtype=GRB.CONTINUOUS, obj=cw,lb=0)    
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
            model.addConstrs(quicksum(HL[0,j,i+ii] for ii in range(24))==quicksum(z_hl[z,hl,j,i] for z in range(Nz)) for hl in range(Nhl)) #assume you get at 8 all load for next day, HL[hl,j,i]
               
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
                             EL[el,j,i] for el in range(Nel))                     # electricity consumed at every load node, EL[el,j,i]
            
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

vogliamo che funzioni per un tot di giorni ad intervallo di ore
ma anche su scala meno fine a lungo periodo
vogliamo evidenziare problematica dei giorni consecutivi cattivi.

Per il report: valutare differenza tra lungo e corto: nei più dettagliati 

facciamo la media e la varianza della matrice di covarianza della copula per valutare se ha senso tenere una unica matrice per tutti i paesi.


#%%

import time
start_time=time.time()

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

d=1

inst=4*24
#inst=365*24
Nel = 7
Ns = 1
Nw = 1
Nhl = 1
Nz = 1
Nfc = 1
czfc = 13#*np.ones([Nz,Nfc]) #random
czhl = 0#*np.ones(Nz,Nhl) #cost matrix Nz x Nhl of cost of transport

#EL=np.zeros([7,5,8760])
#HL=np.zeros([1,5,8760])
#ES=np.zeros([5,8760])
#EW=np.zeros([5,8760])

# variables linked to the NODES:
    
# max PV and turbines per farm
ns = model.addVars(Ns,vtype=GRB.CONTINUOUS, obj=cs,lb=0)
nw = model.addVars(Nw,vtype=GRB.CONTINUOUS, obj=cw,lb=0)    
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
z_fc = model.addVars(product(range(Nz),range(Nfc),range(d),range(inst/24)), obj=czfc) # ASSUMPTION: trucks depart at 8 in the morning, once a day, with predicted H2 industrial load / needed fuel cell storage for the whole day.(32%24==8)*var
z_hl = model.addVars(product(range(Nz),range(Nhl),range(d),range(inst/24)), obj=czhl)
# POSSIBLE ADDITION: connect generator nodes with each other? To avoid congestion?


# constraints not linked to data:
for j in range(d):   
    for i in range(inst-1):
        # electricity at s,w,z,fc,el nodes:
        model.addConstrs(quicksum(s_z[s,z,j,i] for s in range(Ns)) +
                         quicksum(w_z[w,z,j,i] for w in range(Nw)) ==
                         EtH[z,j,i] for z in range(Nz))                    # electricity converted to H2 at every elecrolyzer
        model.addConstrs(quicksum(fc_el[fc,el,j,i] for el in range(Nel)) == 
                         0.044*fhte*HtE[fc,j,i] for fc in range(Nfc))      # electricity converted from H2 at every fuel cell: assumes it all goes to load, no other grid connections
        
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
model.addConstrs(meth[z] <= Meth for z in range(Nz))
model.addConstrs(mhte[fc] <= Mhte for fc in range(Nfc))   
model.addConstrs(nw[w]<=mw for w in range(Nw))                                  #maximum wind turbines at a farm

cons1=model.addConstr(ns[0]>=0)
cons2=model.addConstr(ns[0]>=0)
cons3=model.addConstr(ns[0]>=0)
cons4=model.addConstr(ns[0]>=0)

var_time=time.time()-start_time
print('Setting up model, no data: ',var_time,'s')

#%%

es=ES
ew=EW
el=EL
hl=HL

outputs=[]
VARS=[1000,100,10000,10000]
# iteration on data: groups of 5
for group in range(2):
    
    start_time=time.time()
    
    model.remove(cons1)
    model.remove(cons2)
    model.remove(cons3)
    model.remove(cons4)
    
    ES=es[5*group:5*group+5,:]
    EW=ew[5*group:5*group+5,:]
    EL=el[:,5*group:5*group+5,:]
    HL=hl[:,5*group:5*group+5,:]
      
    for j in range(d):
        for i in range(inst-1):
            cons1=model.addConstrs(quicksum(s_el[s,el,j,i] for s in range(Ns)) + 
                                   quicksum(w_el[w,el,j,i] for w in range(Nw)) + 
                                   quicksum(fc_el[fc,el,j,i] for fc in range(Nfc)) ==
                                   EL[el,j,i] for el in range(Nel))                     # electricity consumed at every load node, EL[el,j,i]
            cons2=model.addConstrs(quicksum(s_el[s,el,j,i] for el in range(Nel)) + 
                                   quicksum(s_z[s,z,j,i] for z in range(Nz)) <= 
                                   ns[s]*ES[j,i] for s in range(Ns))                       # electricity exiting at every PV farm, ES[s,j,i] format?????
            cons3=model.addConstrs(quicksum(w_el[w,el,j,i] for el in range(Nel)) + 
                                   quicksum(w_z[w,z,j,i] for z in range(Nz)) <= 
                                   nw[w]*EW[j,i] for w in range(Nw))                       # electricity exiting at every wind farm
        for i in range(int(inst/24):
            cons4=model.addConstrs(quicksum(HL[0,j,i+ii] for ii in range(24))==quicksum(z_hl[z,hl,j,i] for z in range(Nz)) for hl in range(Nhl)) #assume you get at 8 all load for next day, HL[hl,j,i]
    
    ns[0].Start=VARS[0]
    nw[0].Start=VARS[1]
    nhz[0].Start=VARS[2]
    nhfc[0].Start=VARS[3]
    
    model.optimize()
    
    VARS=[ns[0].X,nw[0].X,nhz[0].X,nhfc[0].X]
    outputs=outputs + [VARS]
    iter_time=time.time()-start_time
    # if model.Status!=2:
    print("Status = {}".format(model.Status))
    print('Time of iteration {}: {}'.format(group,iter_time))




#%%

env = Env(params={'OutputFlag': 0})
model=Model(env=env)
x=model.addVars(product(range(3),range(2)),obj=1)
#c=model.addConstr(x[0]<=4)

model.optimize()


all_vars = model.getVars()
values = model.getAttr("X", all_vars)
names = model.getAttr("VarName", all_vars)
for i in range(len(all_vars)):
    names[i].Start=values[i]

model.addConstrs(x[0,i]>=3 for i in range(2))
x[0,:].Start=[4,3]



model.update()
#c.rhs = 3
#model.optimize()



model.setParam(GRB.Param.TimeLimit, timelimit)
model.setParam(GRB.Param.Method, 2)

model.update()

model.setAttr(GRB.Attr.NumStart,1)
model.setParam(GRB.Param.StartNumber,0))

var.PStart
var.VarHintVal



