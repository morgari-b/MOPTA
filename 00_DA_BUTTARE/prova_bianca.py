# %% First Cell
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env
import time
from itertools import product


# %%

def load_data(path):

    EL=pd.read_excel(path,sheet_name='Electricity Load')
    GL=pd.read_excel(path,sheet_name='Gas Load')
    S=pd.read_excel(path,sheet_name='Solar')
    W=pd.read_excel(path,sheet_name='Wind')

    el=[]
    gl=[]
    s=[]
    w=[]
    for i in range(4):
        el = el + [EL.loc[EL["Quarter"]=="Q{}".format(i+1)].groupby("Instance")["Load"].sum().to_list()]
        gl = gl + [GL.loc[GL["Quarter"]=="Q{}".format(i+1)]["Load"].to_list()]
        s = s + [S.loc[S["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()]
        w = w + [W.loc[W["Quarter"]=="Q{}".format(i+1)]["Generation"].to_list()]
    EL = np.matrix(el)
    HL = np.matrix(gl)
    ES = np.matrix(s)
    EW = np.matrix(w)
    
    return ES, EW, EL, HL
    


# %%
def OPT(ES,EW,EL,HL,cs=4000,cw=3000000,Mns=np.inf,Mnw=100,Mnh=np.inf,ch=7,chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000):
    """
    ES: solar el production - dxinst array
    EW: wind el production - dxinst array
    EL: electricity load - dxinst array
    HL: hydrogen load - dxinst array
    Mns: cost of solar panels
    Mnw: cost of wind turbines
    mw: max socially acceptable wind turbines
    ch: cost of hydrogen storage???
    chte: cost of H to el
    fhte: efficiency of H to el
    Mhte: max H to el at an instance
    ceth: cost of el to H
    feth: efficiency of el to H
    Meth: max el to H at an instance
    
    """
    
    start_time=time.time()
    
    d,inst=np.shape(ES)
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,lb=0) #integer?    
    HtE = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01)

    model.addConstr( ns <= Mns )
    model.addConstr( nw <= Mnw )
    model.addConstr( nh <= Mnh )
    model.addConstr( meth <= Meth )
    model.addConstr( mhte <= Mhte )  
    for j in range(d):
        model.addConstrs( EL[j,i] + EtH[j,i] <= 0.044*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] for i in range(inst)) # put == so no waste: we don't know the future, convert what we have asap
        model.addConstrs( H[j,i+1] == H[j,i] + 28.5*feth*EtH[j,i] - HL[j,i] - HtE[j,i] for i in range(inst-1))
        model.addConstr( H[j,0] == H[j,inst-1] + 28.5*feth*EtH[j,inst-1] - HL[j,inst-1] - HtE[j,inst-1] )  #CIRCULAR
        model.addConstrs( H[j,i] <= nh for i in range(inst))
        model.addConstrs( EtH[j,i] <= meth for i in range(inst) )
        model.addConstrs( HtE[j,i] <= mhte for i in range(inst) )
   
    
    print('Model ready: {}s'.format(time.time()-start_time))
    
    model.optimize()
    if model.Status!=2:
        print("Status = {}".format(model.Status))
        return None, None, model.Status
    else:
        string = "Status: {}\nOptimization time: {}\nTotal cost: {}\nPanels: {}\nTurbines: {}\nH2 needed capacity: {}\nMax EtH: {}\nMax HtE: {}".format(model.Status, time.time()-start_time, model.ObjVal, ns.X,nw.X,nh.X,meth.X,mhte.X)
        print(string)
        time_range = pd.date_range("2024/01/01", periods = ES.shape[1], freq = "h")
        x = time_range
        HH=[0.0033*H[0,i].X for i in range(inst)]
        ETH=pd.DataFrame([EtH[0,i].X for i in range(inst)], index = x)
        HTE=pd.DataFrame([HtE[0,i].X for i in range(inst)], index = x)
        #plt.plot(x,EL[0,:].transpose(),"yellow",x,0.05*HL[0,:].transpose(),"green",x,ns.X*ES.transpose(),"orange",x,nw.X*EW[0,:].transpose(),"red",x,HH,"blue")
        #plt.plot(x,EL[0,:].transpose(),"yellow",x,0.05*HL[0,:].transpose(),"green",x,ns.X*ES.transpose(),"orange",x,nw.X*EW[0,:].transpose(),"red",x,HH,"blue")
        
        #Add date
        
        ES =  pd.DataFrame(ES, columns = time_range)
        EW =  pd.DataFrame(EW, columns = time_range)
        
         # Create a new figure
        fig = plt.Figure(figsize=(8, 7))
        fig.suptitle('Energy System Overview')
        
        # Subplot for Loads
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(x, EL[0, :].T, color="yellow", label="Electricity Load")
        ax1.plot(x, 0.05 * HL[0, :].T, color="green", label="Hydrogen Load")
        ax1.set_xlabel("Date")
        ax1.set_title("Loads  (MWh)")
        ax1.legend()
        
        # Subplot for Loads
        day = 22
        dday = 1
        sx = x[day*24:(day+dday)*24]
        ax1r = fig.add_subplot(3, 2, 2)
        ax1r.plot(sx, EL[0, day*24:(day+dday)*24].T, color="yellow", label="Electricity Load")
        ax1r.plot(sx, 0.05 * HL[0, day*24:(day+dday)*24].T, color="green", label="Hydrogen Load")
        ax1r.set_xlabel("Date")
        ax1r.set_ylabel("Load (MWh)")
        ax1r.set_title("Energy demand during one day")
        ax1r.legend()
        
        # Subplot for Power Output
        ax2 = fig.add_subplot(3, 2, 3)
        
        ax2.plot(x, ns.X * ES.iloc[0, :].T, color="orange", label="Solar Power")
        ax2.plot(x, nw.X * EW.iloc[0, :].T, color="red", label="Wind Power")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Power Output (MWh)")
        ax2.set_title("Power Output during Year")
        ax2.legend()
        
        ax2r= fig.add_subplot(3, 2, 4)
        
        ax2r.plot(sx, ns.X * ES.iloc[0, day*24:(day+dday)*24].T, color="orange", label="Solar Power")
        ax2r.plot(sx, nw.X * EW.iloc[0, day*24:(day+dday)*24].T, color="red", label="Wind Power")
        ax2r.set_xlabel("Date")
        ax2r.set_ylabel("Power Output (MWh)")
        ax2r.set_title("Power Output during one day")
        ax2r.legend()
        
        # Subplot for Stored Hydrogen
        ax3 = fig.add_subplot(3, 2, 5)
        ax3.plot(x, HH, color="blue", label="Stored Hydrogen")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Stored Power (MW)")
        ax3.set_title("Stored Hydrogen during year")
        ax3.legend()
        
        ax3r = fig.add_subplot(3, 2, 6)
        ax3r.plot(sx, HH[day*24:(day+dday)*24], color="blue", label="Stored Hydrogen")
        ax3r.set_xlabel("Date")
        ax3r.set_ylabel("Stored Power (MW)")
        ax3r.set_title("Stored Hydrogen during one day")
        ax3r.legend()
        
        # Display the figure
        #plt.show()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        results = [ns.X,nw.X,nh.X,mhte.X,meth.X, ETH, HTE, HH]
        
        return results, fig, model.Status
    

#%%
 
if __name__ == "__main__":
    
 
    
    ES=0.015*pd.read_csv('scenarios/PV_scenario100.csv',index_col=0).to_numpy()
    EW=4*pd.read_csv('scenarios/wind_scenarios.csv',index_col=0).to_numpy()
    EL=np.zeros([7,100,8760])
    for i,loc in enumerate(['1_i','2_i','3_r','4_r','5_r','1_r','2_r']):
        el=pd.read_csv('scenarios/electric_demand{}.csv'.format(loc),index_col=0).to_numpy()
        EL[i,:,:]=el
    EL=np.sum(EL,0)
    #HL=np.zeros([1,100,8760])
    #HL[0,:,:]=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()
    #%%
    HL=pd.read_csv('scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()
    
    #%% diminish number of scenarios
    n_scenarios = 5
    es = np.array(ES[0:n_scenarios,:])
    ew = np.array(EW[0:n_scenarios,:])
    el = np.array(EL[0:n_scenarios,:])
    hl = np.array(HL[0:n_scenarios,:])
    #%%
    
    """
    self.cs = 4000
    self.cw = 3000000
    self.ch = 10
    self.Mns = np.inf
    self.Mnw = np.inf
    self.Mnh = np.inf
    self.chte = 2
    self.Mhte = np.inf
    self.ceth = 200
    self.feth = 0.7
    self.fhte=0.75
    self.Meth = np.inf
    self.PV_Pmax = 0.015
    self.wind_Pmax = 4
    self.n_scenarios_opt = 5
    """
    
    results0 = OPT(es,ew,el,hl,cs=4000,cw=3000000,Mns=np.inf,Mnw=np.inf,Mnh=np.inf,ch=10,chte=0,fhte=0.75,Mhte=np.inf,ceth=0,feth=0.7,Meth=np.inf)
    #%%
    
    results = results0[0]
    [ns,nw,nh, mhte, meth, ETH, HTE,HH] = results
#%% plots

   
#%%
    # Subplot for Power Output
    fig, ax = plt.subplots(figsize=(10,6))
    x = pd.date_range("01/01/2024", periods = 24*365, freq="h")
    ax.plot(x, ns * ES[0, :].T, color="orange", label="Solar Power")
    ax.plot(x, nw * EW[0, :].T, color="red", label="Wind Power")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power Output (MWh)")
    ax.set_title("Power Output during Year")
    ax.legend()
    plt.savefig("texproject/immagini/ImmaginiOptimization/PowerOutputCETH0")
 #%%   
# Subplot for Stored Hydrogen
    day = 0
    dday = 365
    sx = x[day*24:(day+dday)*24]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sx, HH[day*24:(day+dday)*24], color="blue", label="Stored Hydrogen")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stored Power (MW)")
    ax.set_title("Stored Hydrogen")
    ax.legend()
    plt.savefig("texproject/immagini/ImmaginiOptimization/HHyearCETH0")
        
    #%%
    
    fig, ax = plt.subplots(figsize=(10,6))
    x = pd.date_range("01/01/2024", periods = 24*365, freq="h")
    ax.plot(sx, ns * ES[0, day*24:(day+dday)*24].T, color="orange", label="Solar Power")
    ax.plot(sx, nw * EW[0, day*24:(day+dday)*24].T, color="red", label="Wind Power")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power Output (MWh)")
    ax.set_title("Power Output during a day")
    ax.legend()
    plt.savefig("texproject/immagini/ImmaginiOptimization/PowerOutputdayCETH0")
        
        



