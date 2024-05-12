# %% First Cell
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env

# %%

# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

EL=pd.read_excel('data.xlsx',sheet_name='Electricity Load')
GL=pd.read_excel('data.xlsx',sheet_name='Gas Load')
S=pd.read_excel('data.xlsx',sheet_name='Solar')
W=pd.read_excel('data.xlsx',sheet_name='Wind')

el2=[]
gl2=[]
s2=[]
w2=[]
for i in range(4):
    el2 = el2 + 4*EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby((EL["Instance"]-1)//4)["Load"].sum().to_list()
    gl2 = gl2 + 4*GL.loc[GL["Quarter"]=="Q{}".format(i+1)].groupby((GL["Instance"]-1)//4)["Load"].sum().to_list()
    s2 = s2 + 4*S.loc[S["Quarter"]=="Q{}".format(i+1)].groupby((S["Instance"]-1)//4)["Generation"].sum().to_list()
    # first attempt: wind is trated like others. second attempt: consider wind as it is
    w2 = w2 + 4*W.loc[W["Quarter"]=="Q{}".format(i+1)].groupby((W["Instance"]-1)//4)["Generation"].sum().to_list()
    #w2 = w2 + W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()
el2 = np.matrix(el2)
gl2 = np.matrix(gl2)
s2 = np.matrix(s2)
w2 = np.matrix(w2)


#%%

Electric graph: 
    generators and fuel cells, electrolysers and electric loads
    max generators in a node
    congestion?
Hydrogen graph:
    electrolyzers, gas load and fuel cells
    cost of transport
    congestion


# %%
def OPT(ES,EW,EL,HL,
        #solar el production
        #wind el production
        #electricity load
        #hydrogen load
        cs=4000,cw=3000000,mw=50,ch=7,
        #cost of solar panels
        #cost of wind turbines
        #max socially acceptable wind turbines
        #cost of hydrogen storage
        chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000,
        #cost of H to el
        #efficiency of H to el
        #max H to el at an instance
        #cost of el to H
        #efficiency of el to H
        #max el to H at an instance
        ):
        #option to insert the graph
    
    d,inst=np.shape(ES)
    Nel = EL['Location_Electricity'].nunique() #nodes of electricity load
    Nhl = HL['Location_Gas'].nunique()
    
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
            model.addConstr( EL[j,i] + EtH[j,i] <= 0.044*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] ) # put == so no waste: we don't know the future, convert what we have asap
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
        
        plt.subplot(5,1,1)
        plt.plot(x,EL[0,:].transpose(),"yellow",label = "Electricity Load")
        plt.plot(x,0.05*HL[0,:].transpose(),"green", label = "Hydrogen Load")
        plt.title("Loads")
        plt.legend()
        
        plt.subplot(5,1,2)
        plt.plot(x,ns.X*ES.transpose(),"orange", label = "Solar Power")
        plt.plot(x,nw.X*EW[0,:].transpose(),"red", label = "Wind Power")
        plt.legend()
        plt.title("Power Output")
        
        plt.subplot(5,1,3)
        plt.plot(x,HH,"blue", label = "Stored Hydrogen (Kg?)")
        plt.legend()
        plt.title("Stored Hydrogen")
        
        plt.subplot(5,1,4)
        plt.plot(x,EtH[0,:].X,"green", label = "EtH")
        plt.legend()
        plt.title("H conversion")
    
        plt.subplot(5,1,5)
        plt.plot(x,HtE[0,:].X,"blue", label = "HtE")
        plt.legend()
        plt.title("H conversion")
        return EtH.X, HtE.X
    #[ns.X,nw.X,nh.X,mhte.X,meth.X]

# %%



import pypsa
network = pypsa.Network()
for i in range(3):
    network.add('Bus','Bus_'.format(i), carrier='AC')
    network.add('Line','Line_{}{}'.format(i,i+1),bus0='Bus_{}'.format(i),bus1='Bus_{}'.format(i+1),x=0.1,r=0.01)

network.add("Generator", "Wind", bus="Bus_0", p_set=100, control="PQ")
network.add("Load", "Load", bus="Bus_1", p_set=100)

network.plot()




