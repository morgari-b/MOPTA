#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:11:28 2024

@author: morgari
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env
import time
from itertools import product

#%% DATA


# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

ES=0.015*pd.read_csv('scenarios/PV_scenario100.csv',index_col=0).to_numpy()
EW=4*pd.read_csv('scenarios/wind_scenarios.csv',index_col=0).to_numpy()
EL=np.zeros([7,100,8760])
for i,loc in enumerate(['1_i','2_i','3_r','4_r','5_r','1_r','2_r']):
    el=pd.read_csv('scenarios/electric_demand{}.csv'.format(loc),index_col=0).to_numpy()
    EL[i,:,:]=el
EL=np.sum(EL,0)
#HL=np.zeros([1,100,8760])
#HL[0,:,:]=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()
HL=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()

es=ES[:11,:]
ew=EW[:11,:]
el=EL[:11,:]
hl=HL[:11,:]




#%% MODEL


def OPT(es,ew,el,hl,d=5,rounds=4,cs=4000, cw=3000000,ch=10,Mns=np.inf,Mnw=100,Mnh=np.inf,chte=2,fhte=0.75,Mhte=np.inf,ceth=200,feth=0.7,Meth=np.inf):
            
    start_time=time.time()
    
    D,inst = np.shape(es)
    rounds=min(rounds,D//d)
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('Timelimit',30)
    model.setParam("MIPGap",0.01)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,lb=0) #integer?    
    HtE = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)
    
    model.addConstr( ns <= Mns )
    model.addConstr( nw <= Mnw )
    model.addConstr( nh <= Mnh )
    model.addConstr( meth <= Meth )
    model.addConstr( mhte <= Mhte )  
    
    model.addConstrs( H[j,i] <= nh for i in range(inst) for j in range(d))
    model.addConstrs( EtH[j,i] <= meth for i in range(inst) for j in range(d))
    model.addConstrs( HtE[j,i] <= mhte for i in range(inst) for j in range(d))
    

    outputs=[]
    VARS=[1000,100,10000000]
    
    cons1=model.addConstr(ns>=0)
    cons2=model.addConstr(ns>=0)
    cons3=model.addConstr(ns>=0)
    
    var_time=time.time()-start_time
    print('Model has been set up, this took ',np.round(var_time,4),'s.\nNow starting the optimization. ',rounds,' batches of',d,'scenarios each will be optimized. This should take around 30s per batch.\n')
    
    #model.optimize()
    
    
    for group in range(rounds):
        
        start_time=time.time()
        #model.reset()
        
        model.remove(cons1)
        model.remove(cons2)
        model.remove(cons3)
    
        model.update()
        
        ES=es[d*group:d*group+d,:]
        EW=ew[d*group:d*group+d,:]
        EL=el[d*group:d*group+d,:]
        HL=hl[d*group:d*group+d,:]
    
    
        cons1=model.addConstrs( EL[j,i] + EtH[j,i] <= 0.033*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] for i in range(inst) for j in range(d)) 
        cons2=model.addConstrs( H[j,i+1] == H[j,i] + 30*feth*EtH[j,i] - HL[j,i] - HtE[j,i] for i in range(inst-1) for j in range(d))
        cons3=model.addConstrs( H[j,0] == H[j,inst-1] + 30*feth*EtH[j,inst-1] - HL[j,inst-1] - HtE[j,inst-1] for j in range(d))  #CIRCULAR
        
        ns.VarHintVal=VARS[0]
        nw.VarHintVal=VARS[1]
        nh.VarHintVal=VARS[2]
        
        #print('Model ready: {}s'.format(time.time()-start_time))
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[ns.X,nw.X,nh.X]       
            outputs=outputs + [VARS]
            string = "Round {} of {}:\nOptimal values: {}\nOptimization time: {}\n".format(group+1,rounds,VARS, time.time()-start_time)
            #string=string+"\nTotal cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}\nMax EtH: {}\nMax HtE: {}".format(model.ObjVal, ns.X,nw.X,nh.X,meth.X,mhte.X)
            print(string)
    return outputs
    
        
    
    
    
