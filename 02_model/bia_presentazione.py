#%% import packages
#import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, Env #,quicksum
import time
from itertools import product
#from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter, mdates

print(2)
#%% DATA

# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

ES=0.015*pd.read_csv('MOPTA/scenarios/PV_scenario100.csv',index_col=0).to_numpy()
EW=4*pd.read_csv('MOPTA/scenarios/wind_scenarios.csv',index_col=0).to_numpy()
EL=np.array(100*[(450*pd.read_csv('MOPTA/scenarios/electricity_load.csv',index_col='DateUTC')['BE']).to_list()])
HL=pd.read_csv('MOPTA/scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()

from scenario_generation import read_parameters, SG_weib, SG_beta
GREECE=read_parameters('Greece')
Greece_Wind=SG_weib(100,GREECE['wind'][0],GREECE['wind'][1])
Greece_PV=SG_beta(100, GREECE['PV'][0],GREECE['PV'][1],GREECE['PV'][2],False)
ES_GR=0.015*Greece_PV.to_numpy()
EW_GR=4*Greece_Wind.to_numpy()
EL_GR=np.array(100*[(450*pd.read_csv('MOPTA/scenarios/electricity_load.csv',index_col='DateUTC')['GR']).to_list()])

#el=450*pd.read_csv('MOPTA/scenarios/electricity_load.csv',index_col='DateUTC',parse_dates=True).resample('d').mean()

#%% data with pandas

es=0.015*pd.read_csv('MOPTA/scenarios/PV_scenario100.csv',index_col=0).transpose()
es.index = pd.to_datetime(es.index)
ew=4*pd.read_csv('MOPTA/scenarios/wind_scenarios.csv',index_col=0).transpose()
ew.index = pd.to_datetime(es.index)
el=pd.DataFrame(100*[(450*pd.read_csv('MOPTA/scenarios/electricity_load.csv',index_col='DateUTC')['BE']).to_list()]).transpose()
el.index=pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
hl=pd.read_csv('MOPTA/scenarios/hydrogen_demandg.csv',index_col=0).transpose()
hl.index = pd.to_datetime(es.index)

#%% data by day

ES=0.015*pd.read_csv('MOPTA/scenarios/PV_scenario100.csv',index_col=0).to_numpy()
EW=4*pd.read_csv('MOPTA/scenarios/wind_scenarios.csv',index_col=0).to_numpy()
EL=np.array(100*[(450*pd.read_csv('MOPTA/scenarios/electricity_load.csv',index_col='DateUTC')['BE']).to_list()])
HL=pd.read_csv('MOPTA/scenarios/hydrogen_demandg.csv',index_col=0).to_numpy()


#%% OPT : MIP


def OPT(es,ew,el,hl,d=5,rounds=4,cs=4000, cw=3000000,ch=10,Mns=10*5,Mnw=500,Mnh=109,chte=2,fhte=0.75,Mhte=106,ceth=200,feth=0.7,Meth=10*5):
            
    start_time=time.time()
    
    D,inst = np.shape(es)
    rounds=min(rounds,D//d)
    print("\nSTARTING OPT -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    #model.setParam('Timelimit',45)
    model.setParam("MIPGap",0.05)
    
    ns = model.addVar(vtype=GRB.INTEGER, obj=cs,lb=0,ub=Mns)
    nw = model.addVar(vtype=GRB.INTEGER, obj=cw,lb=0,ub=Mnw)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,ub=Mnh) #integer?    
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01, ub=Mhte)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01,ub=Meth)
    
    HtE = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)

    model.addConstrs( H[j,i] <= nh for i in range(inst) for j in range(d))
    model.addConstrs( EtH[j,i] <= meth for i in range(inst) for j in range(d))
    model.addConstrs( HtE[j,i] <= mhte for i in range(inst) for j in range(d))

    outputs=[]
    HH=[]
    HTE=[]
    ETH=[]
    VARS=[90000,450,10000000,20000,1000]
    cons1=model.addConstr(ns>=0)
    cons2=model.addConstr(ns>=0)
    cons3=model.addConstr(ns>=0)
    
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        
        #start_time=time.time()
        gr_start_time=time.time()

        
        ES=es[d*group:d*group+d,:]
        EW=ew[d*group:d*group+d,:]
        EL=el[d*group:d*group+d,:]
        HL=hl[d*group:d*group+d,:]

        model.remove(cons1)
        model.remove(cons2)
        model.remove(cons3)
            
        ns.VarHintVal=VARS[0]
        nw.VarHintVal=VARS[1]
        nh.VarHintVal=VARS[2]
        mhte.VarHintVal=VARS[3]
        meth.VarHintVal=VARS[4]
        
        cons1=model.addConstrs( EL[j,i] + EtH[j,i] <= 0.033*fhte*HtE[j,i] + ns*ES[j,i] + nw*EW[j,i] for i in range(inst) for j in range(d)) 
        cons2=model.addConstrs( H[j,i+1] == H[j,i] + 30*feth*EtH[j,i] - HL[j,i] - HtE[j,i] for i in range(inst-1) for j in range(d))
        cons3=model.addConstrs( H[j,0] == H[j,inst-1] + 30*feth*EtH[j,inst-1] - HL[j,inst-1] - HtE[j,inst-1] for j in range(d))  #CIRCULAR
        
        model.optimize()
        
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[ns.X,nw.X,nh.X,mhte.X,meth.X]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            hh=[H[0,i].X for i in range(inst)]
            HH=HH + [hh]
            hh=[HtE[0,i].X for i in range(inst)]
            HTE=HTE + [hh]
            hh=[EtH[0,i].X for i in range(inst)]
            ETH=ETH + [hh]

            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE

#%% OPT2: LP, set for dual method


def OPT2(es,ew,el,hl,d=5,rounds=4,cs=4000, cw=3000000,ch=10,Mns=10*5,Mnw=500,Mnh=109,chte=2,fhte=0.75,Mhte=106,ceth=200,feth=0.7,Meth=10*5):
            
    start_time=time.time()
    
    D,inst = np.shape(es)
    rounds=min(rounds,D//d)
    print("\nSTARTING OPT2 -- setting up model for {} batches of {} scenarios.\n".format(rounds,d))
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    
    ns = model.addVar(vtype=GRB.CONTINUOUS, obj=cs,ub=Mns)
    nw = model.addVar(vtype=GRB.CONTINUOUS, obj=cw,ub=Mnw)    
    nh = model.addVar(vtype=GRB.CONTINUOUS, obj=ch,ub=Mnh)   
    mhte=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01, ub=Mhte)
    meth=model.addVar(vtype=GRB.CONTINUOUS,obj=0.01,ub=Meth)
    
    HtE = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg      
    EtH = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst)),vtype=GRB.CONTINUOUS,lb=0)

    model.addConstrs( H[j,i] <= nh for i in range(inst) for j in range(d))
    model.addConstrs( EtH[j,i] <= meth for i in range(inst) for j in range(d))
    model.addConstrs( HtE[j,i] <= mhte for i in range(inst) for j in range(d))

    outputs=[]
    VARS=[]
    cons1=model.addConstrs(nh>=0 for j in range(d) for i in range(inst))
    cons2=model.addConstrs(- H[j,i+1] + H[j,i] + 30*feth*EtH[j,i] - HtE[j,i]==0 for j in range(d) for i in range(inst-1))
    cons3=model.addConstrs(- H[j,0] + H[j,inst-1] + 30*feth*EtH[j,inst-1] - HtE[j,inst-1] == 0 for j in range(d))
    
    print('OPT Model has been set up, this took ',np.round(time.time()-start_time,4),'s.')
    
    for group in range(rounds):
        gr_start_time=time.time()

        ES=es[d*group:d*group+d,:]
        EW=ew[d*group:d*group+d,:]
        EL=el[d*group:d*group+d,:]
        HL=hl[d*group:d*group+d,:]

        model.remove(cons1)
        for j in range(d): 
            for i in range(inst-1):
                cons2[j,i].rhs = HL[j,i]
            cons3[j].rhs  = HL[j,inst-1]
        cons1=model.addConstrs(ns*ES[j,i] + nw*EW[j,i] + 0.033*fhte*HtE[j,i] - EtH[j,i] >= EL[j,i] for j in range(d) for i in range(inst))
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
        else:
            VARS=[np.ceil(ns.X),np.ceil(nw.X),nh.X,mhte.X,meth.X]       
            outputs=outputs + [VARS+[model.ObjVal]] 
            print("Round {} of {} - opt time: {}s.".format(group+1,rounds, np.round(time.time()-gr_start_time,3)))
            
    return outputs#,HH,ETH,HTE


    
#%% Optimize

# OPT - Belgium
out_sing = OPT(ES,EW,EL,HL,d=1,rounds=20)
out_mean=[np.mean([i[n] for i in out_sing[0]]) for n in range(5)]
out_sammen5 = OPT(ES,EW,EL,HL,d=5,rounds=4)
out_sammen10 = OPT(ES,EW,EL,HL,d=10,rounds=2)

# OPT2 - Belgium

out_sing2 = OPT2(ES,EW,EL,HL,d=1,rounds=100)
#out_mean2=[np.mean([i[n] for i in out_sing2]) for n in range(5)]
out_sammen52 = OPT2(ES,EW,EL,HL,d=5,rounds=20)
out_sammen102 = OPT2(ES,EW,EL,HL,d=10,rounds=10)

#%% OPT2 - Greece

out_sing_G = OPT2(ES_GR,EW_GR,EL_GR,HL,d=1,rounds=20)
out_mean_G=[np.mean([i[n] for i in out_sing_G]) for n in range(5)]
out_sammen5_G = OPT2(ES_GR,EW_GR,EL_GR,HL,d=5,rounds=4)
out_sammen10_G = OPT2(ES_GR,EW_GR,EL_GR,HL,d=10,rounds=2)

"""
STARTING OPT2 -- setting up model for 20 batches of 1 scenarios.

OPT Model has been set up, this took  0.5199 s.
Round 1 of 20 - opt time: 2.363s.
Round 2 of 20 - opt time: 2.874s.
Round 3 of 20 - opt time: 2.821s.
Round 4 of 20 - opt time: 1.732s.
Round 5 of 20 - opt time: 2.747s.
Round 6 of 20 - opt time: 1.697s.
Round 7 of 20 - opt time: 10.148s.
Round 8 of 20 - opt time: 2.128s.
Round 9 of 20 - opt time: 1.361s.
Round 10 of 20 - opt time: 2.014s.
Round 11 of 20 - opt time: 2.524s.
Round 12 of 20 - opt time: 1.813s.
Round 13 of 20 - opt time: 1.762s.
Round 14 of 20 - opt time: 2.141s.
Round 15 of 20 - opt time: 2.42s.
Round 16 of 20 - opt time: 1.717s.
Round 17 of 20 - opt time: 1.945s.
Round 18 of 20 - opt time: 2.309s.
Round 19 of 20 - opt time: 2.201s.
Round 20 of 20 - opt time: 2.32s.

STARTING OPT2 -- setting up model for 4 batches of 5 scenarios.

OPT Model has been set up, this took  2.5567 s.
Round 1 of 4 - opt time: 6.587s.
Round 2 of 4 - opt time: 10.466s.
Round 3 of 4 - opt time: 6.456s.
Round 4 of 4 - opt time: 18.56s.

STARTING OPT2 -- setting up model for 2 batches of 10 scenarios.

OPT Model has been set up, this took  6.1471 s.
Round 1 of 2 - opt time: 41.798s.
Round 2 of 2 - opt time: 37.2s.

"""

#%% OPT vs OPT2 running time


"""

STARTING OPT2 -- setting up model for 20 batches of 1 scenarios.

OPT Model has been set up, this took  0.5156 s.
Round 1 of 20 - opt time: 1.926s.
Round 2 of 20 - opt time: 2.084s.
Round 3 of 20 - opt time: 2.507s.
Round 4 of 20 - opt time: 1.81s.
Round 5 of 20 - opt time: 2.259s.
Round 6 of 20 - opt time: 1.351s.
Round 7 of 20 - opt time: 1.93s.
Round 8 of 20 - opt time: 1.713s.
Round 9 of 20 - opt time: 1.928s.
Round 10 of 20 - opt time: 2.851s.
Round 11 of 20 - opt time: 1.675s.
Round 12 of 20 - opt time: 1.309s.
Round 13 of 20 - opt time: 2.777s.
Round 14 of 20 - opt time: 2.586s.
Round 15 of 20 - opt time: 2.081s.
Round 16 of 20 - opt time: 1.875s.
Round 17 of 20 - opt time: 1.605s.
Round 18 of 20 - opt time: 2.251s.
Round 19 of 20 - opt time: 1.648s.
Round 20 of 20 - opt time: 3.039s.

STARTING OPT2 -- setting up model for 4 batches of 5 scenarios.

OPT Model has been set up, this took  2.4778 s.
Round 1 of 4 - opt time: 9.136s.
Round 2 of 4 - opt time: 13.62s.
Round 3 of 4 - opt time: 8.748s.
Round 4 of 4 - opt time: 8.039s.

STARTING OPT2 -- setting up model for 2 batches of 10 scenarios.

OPT Model has been set up, this took  5.9079 s.
Round 1 of 2 - opt time: 10.966s.
Round 2 of 2 - opt time: 18.523s.



Nice, but.... outputflag = 1 gives:

...
Concurrent LP optimizer: primal simplex, dual simplex, and barrier
Showing barrier log only...
...
Barrier solved model in 27 iterations and 1.50 seconds (0.62 work units)
...
Solved with barrier
Iteration    Objective       Primal Inf.    Dual Inf.      Time
     769    2.0824912e+09   0.000000e+00   0.000000e+00      2s

Solved in 769 iterations and 1.90 seconds (1.00 work units)
Optimal objective  2.082491175e+09



Wheras if I force the model.setParam('Method',1) to use the dual:

Set parameter Method to value 1
Gurobi Optimizer version 11.0.0 build v11.0.0rc2 (linux64 - "Ubuntu 24.04 LTS")

CPU model: 13th Gen Intel(R) Core(TM) i5-1340P, instruction set [SSE2|AVX|AVX2]
Thread count: 16 physical cores, 16 logical processors, using up to 16 threads

Optimize a model with 43800 rows, 26285 columns and 117712 nonzeros
Model fingerprint: 0x2554d8d2
Coefficient statistics:
  Matrix range     [6e-04, 2e+01]
  Objective range  [1e-02, 3e+06]
  Bounds range     [5e+02, 1e+09]
  RHS range        [3e+02, 3e+03]
Presolve removed 4999 rows and 4999 columns
Presolve time: 0.18s
Presolved: 38801 rows, 21286 columns, 107028 nonzeros

Iteration    Objective       Primal Inf.    Dual Inf.      Time
       0   -6.3360000e+33   9.567650e+32   6.336000e+03      0s
   27189    1.9132557e+09   1.686111e+06   0.000000e+00      5s
   31903    2.0824912e+09   0.000000e+00   0.000000e+00      8s

Solved in 31903 iterations and 8.48 seconds (12.10 work units)
Optimal objective  2.082491175e+09
    
    
    
"""
#%% Validation functions

def Validate(outputs,ES,EW,EL,HL,verbose=False):
    
    delta=EL-outputs[0]*ES-outputs[1]*EW
    J,I=np.shape(ES)
    H=np.zeros([J,I])
    for i in range(I):
        H[:,i]=H[:,i-1]-HL[:,i-1]+0.7*30*np.minimum(-delta[:,i-1](delta[:,i-1]<0),outputs[4])-1/(0.033*0.75)*delta[:,i-1](delta[:,i-1]>0)
    
    M=np.where((delta>outputs[3]*0.033*0.75).any(axis=1))[0]
    N=np.where(np.max(H,axis=1)-np.min(H,axis=1)-(H[:,-1]-H[:,0])>outputs[2])[0]
    Q=np.where(H[:,-1]<0)[0]
    

    if (len(N)>0) & (verbose==2):
        print('Need more storage in', len(N),'scenarios.\nCurrently nh={}, would need {}.'.format(outputs[2],np.max(np.max(H,axis=1)-np.min(H,axis=1)-H[:,-1]+H[:,0])))
    if (len(Q)>0) & (verbose==2):
        print('Net negative H2 in', len(Q),'scenarios. Worst case needs {}kg stored H2.'.format(-np.min(H[:,-1])))
    if (len(M)>0) & (verbose>0):
        if verbose==2:
            print('Need more EtH capacity in', len(M), 'scenarios. \nCurrently mhte={}, would need {}'.format(outputs[3],np.max(delta)/(0.033*0.75)))
        print('\nFeasibility with current mhte: {}%'.format(np.round((J-len(set(M)|set(N)|set(Q)))/J*100,1)))
        print('Feasibility with increased mhte: {}%'.format(np.round((J-len(set(N)|set(Q)))/J*100,1)))
        
    return [np.round((J-len(set(M)|set(N)|set(Q)))/J*100,1),np.round((J-len(set(N)|set(Q)))/J*100,1)]



def Plot_H(outputs,ES,EW,EL,HL,c='lightskyblue',ax=None,date_range=pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h'),offset=False):
    """
    offset False plots H at level 0 on 1st Jan. 
    offset True plots H at level zero when at miniminum level.
    """
    
    delta=EL-outputs[0]*ES-outputs[1]*EW
    J,I=np.shape(ES)
    H=np.zeros([J,I])
    for i in range(I):
        H[:,i]=H[:,i-1]-HL[:,i-1]+0.7*30*np.minimum(-delta[:,i-1](delta[:,i-1]<0),outputs[4])-1/(0.033*0.75)*delta[:,i-1](delta[:,i-1]>0)

    H=pd.DataFrame(H.transpose()-offset*np.min(H,axis=1))
    H.index=pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
    H.loc[date_range].plot(ax=ax,color=c,ylabel='kg of hydrogen',legend=False,x_compat=True,rot=0)
    return None


def Actual_cost(outputs,ES,EW,EL,HL,cs=4000, cw=3000000,ch=10,chte=2,ceth=200):
    
    delta=EL-outputs[0]*ES-outputs[1]*EW
    J,I=np.shape(ES)
    C=cs*outputs[0]+cw*outputs[1]+ch*outputs[2]+chte/(0.033*0.75*J)np.sum(delta(delta>0))+ceth/J*np.sum(np.minimum(-delta*(delta<0),outputs[4]))
    
    return C

def Gain_H(outputs,ES,EW,EL,HL):
    delta=EL-outputs[0]*ES-outputs[1]*EW
    J,I=np.shape(ES)
    H=np.zeros([J,I])
    for i in range(I):
        H[:,i]=H[:,i-1]-HL[:,i-1]+0.7*30*np.minimum(-delta[:,i-1](delta[:,i-1]<0),outputs[4])-1/(0.033*0.75)*delta[:,i-1](delta[:,i-1]>0)
    
    return H[:,-1]

# On single case, cost is slighly different: don't account for 0.01 cost of mhte and meth to minimize in OPT,
# and at each moment might convert more than strictly necessary (thus higher cost with EtH), but comparable:
    
# for i in range(20):
#     print((Actual_cost(out_sing[0][i],ES[i:i+1],EW[i:i+1],EL[i:i+1],HL[i:i+1])-out_sing[0][i][5])/out_sing[0][i][5])

# Can be used to estimate cost of a batch of scenarios under solution that is not their optimal sol (assuming feasibility - else ignores mhte)

#%% Results for BE

"""
# out_sing                       : average 23.5%, ranging between 2% and 88%
# out_mean                       : 24% 
# out_sammen5                    : average 74%, [59.0, 89.0, 82.0, 66.0]
# out_sammen10                   : 87%, 85%
# np.max(out_sammen5[0],axis=0)  : 93%
# np.max(out_sammen10[0],axis=0) : 93%
# [87867,450,4855887,18547,1029] : 100%

scenarios 24, 41, 47, 52, 77, 78, 84 giving us a hard time


np.mean(np.array([Validate(out_sammen102[i],ES,EW,EL,HL,verbose=0) for i in range(len(out_sammen102))]),axis=0)
Out[1663]: array([82.6, 88.8])

np.mean(np.array([Validate(out_sammen52[i],ES,EW,EL,HL,verbose=0) for i in range(len(out_sammen52))]),axis=0)
Out[1664]: array([69.3 , 80.35])

np.mean(np.array([Validate(out_sing2[i],ES,EW,EL,HL,verbose=0) for i in range(len(out_sing2))]),axis=0)
Out[1665]: array([22.93, 38.15])

Validate(out_mean_sammen102,ES,EW,EL,HL,verbose=0)
Out[1673]: [87.0, 92.0]

Validate(out_mean_sammen52,ES,EW,EL,HL,verbose=0)
Out[1674]: [77.0, 86.0]

Validate(out_mean_sing2,ES,EW,EL,HL,verbose=0)
Out[1675]: [26.0, 47.0]

Validate([np.mean([i[n] for i in out_sammen52[:4]]) for n in range(5)],ES,EW,EL,HL,verbose=0)
Out[1677]: [70.0, 84.0]

Validate([np.mean([i[n] for i in out_sammen102[:2]]) for n in range(5)],ES,EW,EL,HL,verbose=0)
Out[1678]: [79.0, 89.0]

Validate([np.mean([i[n] for i in out_sing2[:20]]) for n in range(5)],ES,EW,EL,HL,verbose=0)
Out[1679]: [22.0, 41.0]

np.mean(np.array([Validate([np.mean([i[n] for i in out_sammen52[j:j+2]]) for n in range(5)],ES,EW,EL,HL,verbose=0) for j in range(10)]),axis=0)
Out[1693]: array([73.8, 84. ])

np.mean(np.array([Validate([np.max([i[n] for i in out_sammen52[j:j+2]]) for n in range(5)],ES,EW,EL,HL,verbose=0) for j in range(10)]),axis=0)
Out[1689]: array([91. , 99.2])

"""
#%% Plot optimal solution of batch

def Sols_plot(out_sing, out_sammen5, out_sammen10,out_mean,title=None):
    fig, ax= plt.subplots(nrows=2,ncols=1,height_ratios=[2,1])
    if title:
        fig.suptitle(title)
    
    ax[0].scatter([i[0] for i in out_sing[0]],[i[1] for i in out_sing[0]],label="single",c='lightskyblue')
    ax[0].scatter(out_mean[0], out_mean[1],label="mean",c='royalblue')
    ax[0].scatter([i[0] for i in out_sammen5[0]],[i[1] for i in out_sammen5[0]],label="joint5",c='lightcoral')
    ax[0].scatter([i[0] for i in out_sammen10[0]],[i[1] for i in out_sammen10[0]],label="joint10",c='firebrick')
    #ax[0].scatter(OUT[0][0],OUT[0][1],label='best',c='green')
    
    ax[0].set_xlabel("solar panels")
    ax[0].set_ylabel("wind turbines")
    ax[0].legend(loc="best")
    
    df=pd.DataFrame(out_sing[0])[[2,3,4]]#.rename(columns={2:'nh',3:'mhte',4:'meth'})
    D=((df-df.mean()).div(df.std())).to_numpy().transpose()
    ax[1].eventplot(D, orientation="horizontal", lineoffsets=[1,2,3], linelength=0.5,colors='lightskyblue')
    
    
    df5=pd.DataFrame(out_sammen5[0])[[2,3,4]]
    D5=((df5-df.mean()).div(df.std())).to_numpy().transpose()
    ax[1].eventplot(D5, orientation="horizontal", lineoffsets=[1,2,3], linelength=0.5, colors='lightcoral')
    
    df10=pd.DataFrame(out_sammen10[0])[[2,3,4]]
    D10=((df10-df.mean()).div(df.std())).to_numpy().transpose()
    ax[1].eventplot(D10, orientation="horizontal", lineoffsets=[1,2,3], linelength=0.5, colors='firebrick')
    
    #dd=pd.DataFrame(OUT[0][2:5]).transpose().rename(columns={0:2,1:3,2:4})
    #DD=((dd-df.mean()).div(df.std())).to_numpy().transpose()
    #ax[1].eventplot(DD,orientation="horizontal", lineoffsets=[1,2,3], linelength=0.5, colors='green')
    
    ax[1].set_yticks([1,2,3],["nh", "mhte", "meth"])
    ax[1].set_xlabel("normalised value")
    
    plt.tight_layout()
    
# Sols_plot([out_sing_G],[out_sammen5_G],[out_sammen10_G],out_mean_G,'Solutions for Greece')

#%% Plot generation, load and H2

def PLOT(outputs,ES,EW,EL,HL,title=None):
    fig = plt.figure(constrained_layout=True,figsize=(6,3))
    subplots = fig.subfigures(2,1, height_ratios=[3, 2])

    ax0 = subplots[0].subplots(2,2,sharex=True)
    ax1 = subplots[1].subplots(1)
    
    
    x=np.linspace(0,12,8760)
    ax0[0,0].plot(x,outputs[0]*ES.transpose(),c='goldenrod')
    ax0[0,0].set_title('ES (MWh)')
    ax0[1,0].plot(x,outputs[1]*EW.transpose(),c='steelblue')
    ax0[1,0].set_title('EW (MWh)')
    ax0[0,1].plot(x,EL.transpose(),c='black')
    ax0[0,1].set_title('EL (MWh)')
    ax0[1,1].plot(x,HL.transpose(),c='red')
    ax0[1,1].set_title('HL (kg)')
    
    
    ax0[1,0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan'],rotation=70)
    ax0[1,1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan'],rotation=70)
        
    Plot_H(outputs,ES,EW,EL,HL,ax=ax1)
    ax1.set_title('H storage (kg)')
    ax1.set_xticks([0,744,1416,3,4,5,6,7,8,9,10,11,12],['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan'],rotation=70)
      
    
    if title:
        fig.suptitle(title)
        
# PLOT(out_greece[0],ES_GR[0:1],EW_GR[0:1],EL_GR[0:1],HL[0:1],'A scenario for Greece: 97084 PV + 421 turbines')
# PLOT(out_sing[0][0],ES[0:1],EW[0:1],EL[0:1],HL[0:1],'A scenario for Belgium: 87328 PV + 451 turbines')


def PLOT_pd(outputs,ES,EW,EL,HL,title=None,date_range=pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')):
    fig = plt.figure(constrained_layout=True,figsize=[8,5])
    subplots = fig.subfigures(2,1, height_ratios=[3, 2])

    ax0 = subplots[0].subplots(2,2,sharex=True)
    ax1 = subplots[1].subplots(1)
    
    #r=0
    date_form = DateFormatter("%d")
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0[0,0].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0[0,1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0[1,0].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0[1,1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax0[0,0].xaxis.set_major_formatter(date_form)
    ax0[0,1].xaxis.set_major_formatter(date_form)
    ax0[1,0].xaxis.set_major_formatter(date_form)
    ax0[1,1].xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_formatter(date_form)

    es=outputs[0]*pd.DataFrame(ES.transpose())
    es.index = pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
    ax0[0,0].plot(es.loc[date_range].index,es.loc[date_range],c='goldenrod')
    #es.loc[date_range].plot(color='goldenrod',ax=ax0[0,0],rot=r,legend=False,x_compat=True)
    ax0[0,0].set_title('ES (MWh)')
    
    ew=outputs[1]*pd.DataFrame(EW.transpose())
    ew.index = pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
    #ew.loc[date_range].plot(color='steelblue',ax=ax0[1,0],rot=r,legend=False,x_compat=True)
    ax0[1,0].plot(ew.loc[date_range].index,ew.loc[date_range],c='steelblue')
    ax0[1,0].set_title('EW (MWh)')
    
    el=pd.DataFrame(EL.transpose())
    el.index = pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
    #el.loc[date_range].plot(color='black',ax=ax0[0,1],rot=r,legend=False,x_compat=True)
    ax0[0,1].plot(el.loc[date_range].index,el.loc[date_range],c='black')
    ax0[0,1].set_title('EL (MWh)')
    
    hl=pd.DataFrame(HL.transpose())
    hl.index = pd.date_range('2023-01-01 00:00:00','2023-12-31 23:00:00',freq='h')
    #hl.loc[date_range].plot(color='red',ax=ax0[1,1],rot=r,legend=False,x_compat=True)
    ax0[1,1].plot(hl.loc[date_range].index,hl.loc[date_range],c='red')
    ax0[1,1].set_title('HL (kg)')
    
    Plot_H(outputs,ES,EW,EL,HL,ax=ax1,date_range=date_range)
    ax1.set_title('H storage (kg)')
    
    if title:
        fig.suptitle(title)


#%% Plot H2 over the year: scenario 1 vs 8 

def H_year():
    fig,ax = plt.subplots(nrows=2)
    fig.suptitle('Two scenarios for Belgium')
    
    Plot_H(out_sing[0][0],ES[0:1],EW[0:1],EL[0:1],HL[0:1],ax=ax[0])
    Plot_H(out_sammen5[0][0],ES[0:1],EW[0:1],EL[0:1],HL[0:1], c='lightcoral',ax=ax[0])
    Plot_H(out_sammen10[0][0],ES[0:1],EW[0:1],EL[0:1],HL[0:1], c='firebrick',ax=ax[0])
    
    ax[0].legend(['sing','joint5','joint10'])
    ax[0].set_title('Scenario from training set')
    
    Plot_H(out_sing[0][18],ES[18:19],EW[18:19],EL[18:19],HL[18:19],ax=ax[1],c='black')
    Plot_H(out_sing[0][7],ES[18:19],EW[18:19],EL[18:19],HL[18:19],ax=ax[1])
    Plot_H(out_sammen5[0][0],ES[18:19],EW[18:19],EL[18:19],HL[18:19], c='lightcoral',ax=ax[1])
    Plot_H(out_sammen10[0][0],ES[18:19],EW[18:19],EL[18:19],HL[18:19], c='firebrick',ax=ax[1])
    
    ax[1].legend(['best fit'],loc='lower center')
    ax[1].set_title('Scenario from test set')
    
    plt.tight_layout()


# Train: the fit goes to 0 at the end, circular. 
#        the robust ones have more storage and meth than strictly necessary for scenario sing,
#        so they accumulate during the year (with higher cost)

# Test : single isn't feasible on a new scenario: go to negative H2. 
#        sammen are feasible and gain lots of storage

# how much storage?


    

#%% costs

print(Actual_cost(out_sing[0][8],ES[8:9],EW[8:9],EL[8:9],HL[8:9]),
      Actual_cost(out_sammen5[0][1],ES[4:9],EW[4:9],EL[4:9],HL[4:9]),
      Actual_cost(out_sammen10[0][0],ES[:9],EW[:9],EL[:9],HL[:9])   )

print(Actual_cost(out_sing[0][8],ES[8:9],EW[8:9],EL[8:9],HL[8:9]),
      Actual_cost(out_sammen5[0][1],ES[8:9],EW[8:9],EL[8:9],HL[8:9]),
      Actual_cost(out_sammen10[0][0],ES[8:9],EW[8:9],EL[8:9],HL[8:9])   )

"""
np.mean([i[5] for i in out_sing[0]])
Out[508]: 2065924722.5748954

np.mean([i[5] for i in out_sammen5[0]])
Out[509]: 2083578537.8540273

np.mean([i[5] for i in out_sammen10[0]])
Out[510]: 2089325757.0637321

"""

#%% Multiple  years

def Multiple_years():
    C=[4000,3000000,10]
    outputs=[]
    
    years = [1,2,5,10,20,50]
    for i in years:
        out=OPT2(ES,EW,EL,HL,1,1,C[0]/i,C[1]/i,C[2]/i,Mnw=GRB.INFINITY)[0]
        outputs+=[out]
        
    
    OUT=pd.DataFrame(outputs)
    OUT['year']=years
    OUT.set_index('year',inplace=True)
    OUT.rename(columns={0:'ns',1:'nw',2:'nh',3:'mhte',4:'meth',5:'objVal'},inplace=True)
    
    df=OUT[['ns','nw','nh','objVal']].div(OUT[['ns','nw','nh','objVal']].max())
    df.plot(xticks=years, legend=False, color=['goldenrod','steelblue','green','firebrick'])
    plt.legend(['ns','nw','nh','cost'],loc='upper right')
    plt.title('Optimal solutions long term')
    
    
# Over longer periods the marginal costs become more relevant, thus it is convenient to use less hydrogen and rely on greater availability of generation thanks to bigger upfront capital investment.
# Less hydrogen means less ability to redistribute energy over the seasons: summer solar production gets lost, it's better to invest in wind.


#%%





OUT = OPT2(ES,EW,EL,HL,d=100,rounds=1)


#%%


opt_5m=[[np.max([i[n] for i in out_sammen52[j:j+2]]) for n in range(5)] for j in range(10)]

Cost_5m=np.mean([Actual_cost(opt_5m[j],ES,EW,EL,HL) for j in range(10)])

Cost_OUT=Actual_cost(OUT[0],ES,EW,EL,HL)

Cost_sing_av=np.mean([Actual_cost(out_sing2[i],ES[i:i+1],EW[i:i+1],EL[i:i+1],HL[i:i+1]) for i in range(99)])


"""
Cost_5m
Out[1753]: 2140697348.5609875

Cost_OUT
Out[1754]: 2132797829.3219507


Cost_sing_av
Out[1758]: 2067136982.0187514

"""







np.max(np.array([out_sammen10[0][i] for i in range(2)]),axis=0)

Actual_cost(_,ES,EW,EL,HL)
Validate(_,ES,EW,EL,HL)

"""

Validate(np.max(np.array([out_sammen10[0][i] for i in range(2)]),axis=0),ES,EW,EL,HL,verbose=1)
Need more HtE capacity or higher production in scenarios: [24 41 47 52 77 78 84] . 
Currently mhte=15627.908514417708, would need 18169.90889379465
Feasibility: 93.0%
Out[1310]: 93.0

Validate(np.max(np.array([out_sammen5[0][i] for i in range(4)]),axis=0),ES,EW,EL,HL,verbose=1)
Need more HtE capacity or higher production in scenarios: [24 41 47 52 77 78 84] . 
Currently mhte=15774.505087556317, would need 18182.733849003125
Feasibility: 93.0%
Out[1311]: 93.0

Validate(np.mean(np.array([out_sammen5[0][i] for i in range(4)]),axis=0),ES,EW,EL,HL,verbose=1)
Need more HtE capacity or higher production in scenarios: [ 8 10 15 24 41 45 47 52 55 67 68 72 77 78 84 91] . 
Currently mhte=15323.133560540038, would need 18282.127251868784
Need more storage in scenarios:  [ 8 13 25 30 37 45 61 98 99] .
Currently nh=3893753.3000551355, would need 4897852.416187716.
Feasibility: 77.0%
Out[1312]: 77.0

Validate(np.mean(np.array([out_sammen10[0][i] for i in range(2)]),axis=0),ES,EW,EL,HL,verbose=1)
Need more HtE capacity or higher production in scenarios: [15 24 41 47 52 55 77 78 84 91] . 
Currently mhte=15576.695794527186, would need 18221.20871462854
Need more storage in scenarios:  [30 37 45 61] .
Currently nh=4282114.995836785, would need 4787756.908154206.
Feasibility: 86.0%
Out[1313]: 86.0



Validate(out_sammen5[0][0],ES,EW,EL,HL,verbose=1)

Feasibility with current mhte: 59.0%
Feasibility with increased mhte: 85.0%
Out[1347]: 59.0

Validate(out_sammen5[0][1],ES,EW,EL,HL,verbose=1)

Feasibility with current mhte: 89.0%
Feasibility with increased mhte: 99.0%
Out[1348]: 89.0

Validate(out_sammen5[0][2],ES,EW,EL,HL,verbose=1)

Feasibility with current mhte: 82.0%
Feasibility with increased mhte: 94.0%
Out[1349]: 82.0

Validate(out_sammen5[0][3],ES,EW,EL,HL,verbose=1)

Feasibility with current mhte: 66.0%
Feasibility with increased mhte: 74.0%
Out[1350]: 66.0


ES[[24, 41, 47, 52, 77, 78, 84]],EW[[24 41 47 52 77 78 84]],EL[[24 41 47 52 77 78 84]],HL[[24 41 47 52 77 78 84]]












"""