# %% First Cell

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, quicksum, Env

# %%

EL=pd.read_excel('data.xlsx',sheet_name='Electricity Load')
GL=pd.read_excel('data.xlsx',sheet_name='Gas Load')
S=pd.read_excel('data.xlsx',sheet_name='Solar')
W=pd.read_excel('data.xlsx',sheet_name='Wind')

# prepare data into clean single days...
# potentially start at midday?
el=[]
gl=[]
s=[]
w=[]
for i in range(4):
    el = el + [EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby("Instance")["Load"].sum().to_list()]
    gl = gl + [GL.loc[GL["Quarter"]=="Q{}".format(i+1)]["Load"].tolist()]
    s = s + [S.loc[S["Quarter"]=="Q{}".format(i+1)]["Generation"].tolist()]
    w = w + [W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].tolist()]
el = np.matrix(el)
gl = np.matrix(gl)
s = np.matrix(s)
w = np.matrix(w)

# %%
def OPT(ES,EW,EL,HL,cs=10000,cw=1000000,ch=10000,chte=0,fhte=1,mhte=10,ceth=0,feth=1,meth=10):
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
    
    d,inst=np.shape(ES)
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,lb=0) #integer?    
    HtE = model.addMVar((d,inst),vtype=GRB.CONTINUOUS, obj=chte,lb=0) # expressed in kg      
    EtH = model.addMVar((d,inst),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in power
    H = model.addMVar((d,inst),vtype=GRB.CONTINUOUS,lb=0)

    for j in range(d):
        for i in range(inst):
            model.addConstr( EL[j,i] + EtH[j,i] <= fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] )
            model.addConstr( H[j,i] == H[j,i-1] + feth*EtH[j,i] - HL[j,i] - HtE[j,i] )
            model.addConstr( H[j,i] <= nh )
    
    model.optimize()
    
    string = "Status: {}\nTotal cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}".format(model.Status, model.ObjVal,ns.X,nw.X,nh.X)
    return print(string)

# %%
el1 = np.matrix(EL.loc[(EL["Quarter"]=="Q2") & (EL["Instance"] >= 48)].groupby("Instance")["Load"].sum().to_list())
gl1 = np.matrix(GL.loc[(GL["Quarter"]=="Q2") & (GL["Instance"] >= 48)]["Load"].tolist())
s1 = np.matrix(S.loc[(S["Quarter"]=="Q2") & (S["Instance"] >= 48)]["Generation"].tolist())
w1 = np.matrix(W.loc[(W["Quarter"]=="Q2") & (W["Instance"] >= 48)]["Generation"].tolist())

# %%

def OPT1(ES,EW,EL,HL,cs=10000,cw=1000000,ch=10000,chte=0,fhte=1,mhte=10,ceth=0,feth=1,meth=10):
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


# %%

fig, ax = plt.subplots(1,1)
x=np.linspace(0,24,96)
plt.plot(x,el[0,:].transpose()+gl[0,:].transpose(),x,100000*s[0].transpose(),x,200*w[0].transpose())

