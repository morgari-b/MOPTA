#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:43:04 2024

@author: frulcino
"""
#this file contains the functions called by the UI
# %% First Cell
import os
# Change the current working directory
os.chdir('/home/frulcino/codes/MOPTA/')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from gurobipy import Model, GRB, quicksum, Env
from matplotlib.figure import Figure



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
        
        # Create a new figure
        fig = plt.Figure(figsize=(8, 7))
        fig.suptitle('Energy System Overview')
        
        # Subplot for Loads
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(x, EL[0, :].T, color="yellow", label="Electricity Load")
        ax1.plot(x, 0.05 * HL[0, :].T, color="green", label="Hydrogen Load")
        ax1.set_title("Loads")
        ax1.legend()
        
        # Subplot for Power Output
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(x, ns.X * ES[0, :].T, color="orange", label="Solar Power")
        ax2.plot(x, nw.X * EW[0, :].T, color="red", label="Wind Power")
        ax2.set_title("Power Output")
        ax2.legend()
        
        # Subplot for Stored Hydrogen
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(x, HH, color="blue", label="Stored Hydrogen (Kg?)")
        ax3.set_title("Stored Hydrogen")
        ax3.legend()
        
        # Display the figure
        plt.show()
        return (model.ObjVal, fig) #[ns.X,nw.X,nh.X,mhte.X,meth.X]
    
def run_OPT(cs=4000,cw=3000000,mw=100,ch=10000,chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000):
    ES, EW, EL, HL = load_data("data.xlsx")
    objval, fig = OPT(ES,EW,EL,HL,cs,cw,mw,ch,chte,fhte,Mhte,ceth,feth,Meth)
    print(objval)
    return objval, fig