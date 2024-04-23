# %% First Cell

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env

# %%

EL=pd.read_excel('data.xlsx',sheet_name='Electricity Load')
GL=pd.read_excel('data.xlsx',sheet_name='Gas Load')
S=pd.read_excel('data.xlsx',sheet_name='Solar')
W=pd.read_excel('data.xlsx',sheet_name='Wind')

# prepare data into clean single days...
# potentially start at midday?
el = np.matrix(EL.loc[EL["Quarter"]=="Q1"].groupby("Instance")["Load"].sum().to_list())
gl = np.matrix(GL.loc[GL["Quarter"]=="Q1"]["Load"].tolist())
s = np.matrix(S.loc[S["Quarter"]=="Q1"]["Generation"].tolist())
w = np.matrix(W.loc[W["Quarter"]=="Q1"]["Generation"].tolist())

# %%
def OPT(ES,EW,EL,HL,cs=10000,cw=1000000,ch=10,chte=0,fhte=1,mhte=10,ceth=0,feth=1,meth=10):
    #cost of solar panels
    #cost of wind turbines
    #cost of hydrogen storage???
    #cost of H to el
    #efficiency of H to el
    #max H to el at an instance
    #cost of el to H
    #efficiency of el to H
    #max el to H at an instance
    #solar el production - dx96 array
    #wind el production - dx96 array
    #electricity load - dx96 array
    #hydrogen load - dx96 array
    
    d=np.size(ES)
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,lb=0) #integer?    
    HtE = model.addMVar((d,96),vtype=GRB.CONTINUOUS, obj=chte,lb=0) # expressed in kg      
    EtH = model.addMVar((d,96),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in power
    H = model.addMVar((d,96),vtype=GRB.CONTINUOUS,lb=0)

    for i in range(96):
        #model.addConstr(np.eye(d),EL[:,i] + EtH[:,i], < , fhte*HtE[:,i] + ns*ES[:,i] + nw*EW[:,i] )
        #model.addMConstr( H[:,i] = H[:,i-1] + feth*EtH[:,i] - HL[:,i] - HtE[:,i] )
        #model.addConstr(np.eye(d), H[:,i],<,nh*np.ones((d,1)) )
        model.addConstr( EL[:,i] + EtH[:,i] <= fhte*HtE[:,i] + ns*ES[:,i] + nw*EW[:,i] )
    
    model.optimize()
    
    string = "Total cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}".format(model.ObjVal,ns.X,nw.X,nh.X)
    return print(string)

# %%
el = EL.loc[EL["Quarter"]=="Q1"].groupby("Instance")["Load"].sum().to_list()
gl = GL.loc[GL["Quarter"]=="Q1"]["Load"].tolist()
s = S.loc[S["Quarter"]=="Q1"]["Generation"].tolist()
w = W.loc[W["Quarter"]=="Q1"]["Generation"].tolist()

def OPT1(ES,EW,EL,HL,cs=10000,cw=1000000,ch=10,chte=0,fhte=1,mhte=10,ceth=0,feth=1,meth=10):
    #cost of solar panels
    #cost of wind turbines
    #cost of hydrogen storage???
    #cost of H to el
    #efficiency of H to el
    #max H to el at an instance
    #cost of el to H
    #efficiency of el to H
    #max el to H at an instance
    #solar el production - dx96 array
    #wind el production - dx96 array
    #electricity load - dx96 array
    #hydrogen load - dx96 array
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,lb=0) #integer?    
    HtE = [model.addVar(vtype=GRB.CONTINUOUS, obj=chte,lb=0) for i in range(96)] # expressed in kg      
    EtH = [model.addVar(vtype=GRB.CONTINUOUS, obj=ceth, lb=0) for i in range(96)] # expressed in power
    H = [model.addVar(vtype=GRB.CONTINUOUS,lb=0) for i in range(96)]

    for i in range(96):
        model.addConstr( EL[i] + EtH[i] <= fhte*HtE[i] + ns*ES[i] + nw*EW[i] )
        model.addConstr( H[i] == H[i-1] + feth*EtH[i] - HL[i] - HtE[i] )
        model.addConstr( H[i] <= nh )
    
    model.optimize()
    
    string = "Total cost: {} \nPanels: {} \nTurbines: {} \nH2 needed capacity: {}".format(model.ObjVal,ns.X,nw.X,nh.X)
    return print(string)


