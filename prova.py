# %% First Cell
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env

# %%

# Change the current working directory
os.chdir('/home/frulcino/codes/MOPTA/')

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
    gl = gl + [GL.loc[GL["Quarter"]=="Q{}".format(i+1)]["Load"].to_list()]
    s = s + [S.loc[S["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()]
    w = w + [W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()]
el = np.matrix(el)
gl = np.matrix(gl)
s = np.matrix(s)
w = np.matrix(w)

# %%
def OPT(ES,EW,EL,HL,cs=4000,cw=3000000,mw=100,ch=10000,chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000):
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
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)

    for j in range(d):
        for i in range(inst):
            model.addConstr( EL[j,i] + EtH[j,i] == 0.044*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] ) # put == so no waste: we don't know the future, convert what we have asap
            model.addConstr( H[j,i] == H[j,i-1] + 28.5*feth*EtH[j,i] - HL[j,i] - HtE[j,i] )
            model.addConstr( H[j,i] <= nh )
            model.addConstr( EtH[j,i] <= meth)
            model.addConstr( HtE[j,i] <= mhte)
    model.addConstr( nw <= mw )
    model.addConstr( meth <= Meth )
    model.addConstr( mhte <= Mhte )     
    
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
        return None
    else:
        string = "Status: {}\nTotal cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}\nMax EtH: {}\nMax HtE: {}".format(model.Status, model.ObjVal, ns.X,nw.X,nh.X,meth.X,mhte.X)
        print(string)
        x=np.arange(inst)
        HH=[0.0033*H[0,i].X for i in range(inst)]
        #plt.plot(x,EL[0,:].transpose(),"yellow",x,0.05*HL[0,:].transpose(),"green",x,ns.X*ES.transpose(),"orange",x,nw.X*EW[0,:].transpose(),"red",x,HH,"blue")
        #plt.plot(x,EL[0,:].transpose(),"yellow",x,0.05*HL[0,:].transpose(),"green",x,ns.X*ES.transpose(),"orange",x,nw.X*EW[0,:].transpose(),"red",x,HH,"blue")
        
        #Solar panels and wind turbines
        
        plt.subplot(3,1,1)
        plt.plot(x,EL[0,:].transpose(),"yellow",label = "Electricity Load")
        plt.plot(x,0.05*HL[0,:].transpose(),"green", label = "Hydrogen Load")
        plt.title("Loads")
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.plot(x,ns.X*ES.transpose(),"orange", label = "Solar Power")
        plt.plot(x,nw.X*EW[0,:].transpose(),"red", label = "Wind Power")
        plt.legend()
        plt.title("Power Output")
        
        plt.subplot(3,1,3)
        plt.plot(x,HH,"blue", label = "Stored Hydrogen (Kg?)")
        plt.legend()
        plt.title("Stored Hydrogen")
        return model.ObjVal #[ns.X,nw.X,nh.X,mhte.X,meth.X]

# %%

def VarsToArray(VecVar):
    # VecVar: A list of Gurobi Variables plt.subplot(3,1,1 plt.subplot(3,1,1)
        plt.plot(x,EL[0,:].transpose(),"yellow",label = "Electricity Load")
        plt.plot(x,0.05*HL[0,:].transpose(),"green", label = "Hydrogen Load")
        plt.title("Loads")
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.plot(x,ns.X*ES.transpose(),"orange", label = "Solar Power")
        plt.plot(x,nw.X*EW[0,:].transpose(),"red", label = "Wind Power")
        plt.legend()
        plt.title("Power Output")
        
        plt.subplot(3,1,3)
        plt.plot(x,HH,"blue", label = "Stored Hydrogen (Kg?)")
        plt.legend()
        plt.title("Stored Hydrogen"))
        plt.plot(x,EL[0,:].transpose(),"yellow",label = "Electricity Load")
        plt.plot(x,0.05*HL[0,:].transpose(),"green", label = "Hydrogen Load")
        plt.title("Loads")
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.plot(x,ns.X*ES.transpose(),"orange", label = "Solar Power")
        plt.plot(x,nw.X*EW[0,:].transpose(),"red", label = "Wind Power")
        plt.legend()
        plt.title("Power Output")
        
        plt.subplot(3,1,3)
        plt.plot(x,HH,"blue", label = "Stored Hydrogen (Kg?)")
        plt.legend()
        plt.title("Stored Hydrogen")
    X = VecVar.X
    M = np.zeros(X.shape)
    
    return "miao"

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
gl1 = np.matrix(GL.loc[(GL["Quarter"]=="Q2") & (GL["Instance"] >= 48)]["Load"].to_list())
s1 = np.matrix(S.loc[(S["Quarter"]=="Q2") & (S["Instance"] >= 0.009)]["Generation"].to_list())
w1 = np.matrix(W.loc[(W["Quarter"]=="Q2") & (W["Instance"] >= 48)]["Generation"].to_list())

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

OPT(s1,w1,el1,gl1)

m = OPT2(s1,w1,el1,gl1)
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


#%%

# Suppose we have more than 4 consecutive days: aggregate data by hour

el2=[]
gl2=[]
s2=[]
w2=[]
for i in range(4):
    el2 = el2 + 4*EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby((EL["Instance"]-1)//4)["Load"].sum().to_list()
    gl2 = gl2 + 4*GL.loc[GL["Quarter"]=="Q{}".format(i+1)].groupby((GL["Instance"]-1)//4)["Load"].sum().to_list()
    s2 = s2 + 4*S.loc[S["Quarter"]=="Q{}".format(i+1)].groupby((S["Instance"]-1)//4)["Generation"].sum().to_list()
    # first attempt: wind is trated like others. second attempt: consider wind as it is
    #w2 = w2 + 4*W.loc[W["Quarter"]=="Q{}".format(i+1)].groupby((W["Instance"]-1)//4)["Generation"].sum().to_list()
    w2 = w2 + W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()
el2 = np.matrix(el2)
gl2 = np.matrix(gl2)
s2 = np.matrix(s2)
w2 = np.matrix(w2)

fig, ax = plt.subplots(1,1)
x=np.linspace(0,1,384)
plt.plot(x,el2.transpose(),"yellow",x,0.033*gl2.transpose(),"green",x,38173*s2.transpose(),"orange",x,50*w2.transpose(),"blue")


#OPT(s2,w2,el2,gl2)

#Status: 2
#Total cost: 568402068.2360864
#Panels: 38173.0
#Turbines: 50.0
#H2 needed capacity: 26571.000063605356
#Max EtH: 481.16767793471075
#Max HtE: 6278.835611784831

# so.... aggregating by hour doesn't really change much (compare with first example)
# most variability is between seasons and day/night
# but if I treat wind as variable problem becomes unfeasable within contraints
# aka if I don't have wind for two days I'm soup

#%%

el3=[]
gl3=[]
s3=[]
w3=[]
for i in range(4):
    el3 = el3 + EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby((EL["Instance"]-1)//4)["Load"].sum().to_list()
    gl3 = gl3 + GL.loc[GL["Quarter"]=="Q{}".format(i+1)].groupby((GL["Instance"]-1)//4)["Load"].sum().to_list()
    s3 = s3 + S.loc[S["Quarter"]=="Q{}".format(i+1)].groupby((S["Instance"]-1)//4)["Generation"].sum().to_list()
    #w3 = w3 + W.loc[W["Quarter"]=="Q{}".format(i+1)].groupby((W["Instance"]-1)//4)["Generation"].sum().to_list()
w3 = W.loc[W["Quarter"]=="Q1"]["Generation"]
el3 = np.matrix(el3)
gl3 = np.matrix(gl3)
s3 = np.matrix(s3)
w3 = np.matrix(w3)

#fig, ax = plt.subplots(1,1)
#x=np.linspace(0,1,96)
#plt.plot(x,el3.transpose(),"yellow",x,0.033*gl3.transpose(),"green",x,38173*s3.transpose(),"orange",x,50*w3.transpose(),"blue")

costs=[]
for mw in [50,75,100,150,200]:
    costs += [OPT(s3,w3,el3,gl3,mw=mw)]
print(costs)
plt.plot([50,75,100,150,200], costs)

#%%
Ws = W.sort_values(by="Generation").reset_index()["Generation"].to_list()

# observation: wind is modelled through weibull distribution. Look for realistic parameters
Ys = np.sort(2.9*np.random.weibull(3.5,384))
#Ys = [4/(1+math.exp(-x+2)) for x in Ys]  # model using sigmoid activation function
Ys[:] = [4 if x>4 else x for x in Ys]

Xs = np.linspace(0,1,384)
plt.plot(Xs,Ys,"blue",Xs,Ws,"green")

#%% QUESTION: Seasonal variance?
Ys = np.sort(2.9*np.random.weibull(3.5,96))
#Ys = [5/(1+math.exp(-x+3)) for x in Ys]
Ys[:] = [4 if x>4 else x for x in Ys]

plt.plot(np.linspace(0,24,96), np.sort(w[0]).transpose(),"blue",
         np.linspace(0,24,96), np.sort(w[1]).transpose(),"green",
         np.linspace(0,24,96), np.sort(w[2]).transpose(),"yellow",
         np.linspace(0,24,96), np.sort(w[3]).transpose(),"brown",
         
         np.linspace(0,24,96), Ys,"red")

#%%
for i in np.linspace(4,9,4):
    plt.plot(np.linspace(0,24,96), np.sort(np.random.weibull(i,96)))











