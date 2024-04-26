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
def OPT(ES,EW,EL,HL,cs=4000,cw=3000000,mw=50,ch=10000,chte=0,fhte=0.75,mhte=10,ceth=0,feth=0.7,meth=10):
    #cost of solar panels
    #cost of wind turbines
    #max socially acceptable wind turbines
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
    EtH = model.addMVar((d,inst),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in MWh
    H = model.addMVar((d,inst),vtype=GRB.CONTINUOUS,lb=0)

    for j in range(d):
        for i in range(inst):
            model.addConstr( EL[j,i] + EtH[j,i] == 0.044*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] ) # put == so no waste: we don't know the future, convert what we have asap
            model.addConstr( H[j,i] == H[j,i-1] + 28.5*feth*EtH[j,i] - HL[j,i] - HtE[j,i] )
            model.addConstr( H[j,i] <= nh )
    model.addConstr( nw <= mw )        
    
    model.optimize()
    
    string = "Status: {}\nTotal cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}".format(model.Status, model.ObjVal,ns.X,nw.X,nh.X)
    print(string)
    x=np.linspace(0,1,inst)
    HH=[0.0033*H[0,i].X for i in range(inst)]
    plt.plot(x,EL[0,:].transpose(),"yellow",x,0.05*HL[0,:].transpose(),"green",x,ns.X*ES.transpose(),"orange",x,nw.X*EW[0,:].transpose(),"red",x,HH,"blue")

    return [ns.X,nw.X,nh.X]

# %%

fig, ax = plt.subplots(1,1)
x=np.linspace(0,24,96)
plt.plot(x,el.transpose(),"yellow",x,0.033*gl.transpose(),"green",x,5791*s.transpose(),"orange",x,83*w.transpose(),"blue")

# %%
x=np.linspace(0,1,np.shape(w1)[1])
plt.plot(x,el1.transpose()+0.05*gl1.transpose(),"red",x,5791*s1.transpose()+83*w1.transpose(),"green")

# why are dimensions so different??

# %% STARTING AT MIDDAY

el1 = np.matrix(EL.loc[(EL["Quarter"]=="Q2") & (EL["Instance"] >= 48)].groupby("Instance")["Load"].sum().to_list())
gl1 = np.matrix(GL.loc[(GL["Quarter"]=="Q2") & (GL["Instance"] >= 48)]["Load"].tolist())
s1 = np.matrix(S.loc[(S["Quarter"]=="Q2") & (S["Instance"] >= 0.009)]["Generation"].tolist())
w1 = np.matrix(W.loc[(W["Quarter"]=="Q2") & (W["Instance"] >= 48)]["Generation"].tolist())

# %%

# Example of 4 consecutive days

el1=[]
gl1=[]
s1=[]
w1=[]
for i in range(4):
    el1 = el1 + EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby("Instance")["Load"].sum().to_list()
    gl1 = gl1 + GL.loc[GL["Quarter"]=="Q{}".format(i+1)]["Load"].tolist()
    s1 = s1 + S.loc[S["Quarter"]=="Q{}".format(i+1)]["Generation"].tolist()
    w1 = w1 + W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].tolist()
el1 = np.matrix(el1)
gl1 = np.matrix(gl1)
s1 = np.matrix(s1)
w1 = np.matrix(w1)

#OPT(s1,w1,el1,gl1)

#Status: 2
#Total cost: 578375869.0538827
#Panels: 38276.0
#Turbines: 50.0
#H2 needed capacity: 27527.186905388273

# %%

# Example of 4 consecutive days: decreasing generation, same consumption

el1=[]
gl1=[]
s1=[]
w1=[]
for i in range(4):
    el1 = el1 + EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby("Instance")["Load"].sum().to_list()
    gl1 = gl1 + GL.loc[GL["Quarter"]=="Q{}".format(i+1)]["Load"].tolist()
    s1 = s1 + ((3-i)/9*S.loc[S["Quarter"]=="Q{}".format(i+1)]["Generation"]).tolist()
    w1 = w1 + ((3-i)/9*W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"]).tolist()
el1 = np.matrix(el1)
gl1 = np.matrix(gl1)
s1 = np.matrix(s1)
w1 = np.matrix(w1)

#OPT(s1,w1,el1,gl1)

#Status: 2
#Total cost: 7247484221.179818
#Panels: 446436.0
#Turbines: 50.0
#H2 needed capacity: 531174.0221179818
