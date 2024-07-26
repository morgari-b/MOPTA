#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:17:17 2024

@author: morgari


VERIFICA CHE IL VECCHIO MODELLO SU NODO SINGOLO E IL NUOVO MODELLO A PIÙ NODI APPLICATO AD UN NODO SINGOLO COINCIDANO. RISPOSTA: SÌ COINCIDONO.


"""

#%% import modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates

import xarray as xr
import folium

from YUPPY import OPT1, OPT2, Network

#%% prepare data for both methods

# data for OPT1

ES=0.015*pd.read_csv('MOPTA/01_scenario_generation/scenarios/PV_scenario100.csv',index_col=0).head(1)
EW=4*pd.read_csv('MOPTA/01_scenario_generation/scenarios/wind_scenarios.csv',index_col=0).head(1)
EL=pd.DataFrame(450*pd.read_csv('MOPTA/01_scenario_generation/scenarios/electricity_load_2023.csv',index_col='DateUTC')['BE']).T
HL=pd.read_csv('MOPTA/01_scenario_generation/scenarios/hydrogen_demandg.csv',index_col=0).head(1)

# data for OPT2

EU=pd.DataFrame([['Italy',41.90,12.48]], columns=['node','lat','long']).set_index('node')
EU_e=pd.DataFrame(columns=['start_node','end_node']) #empty because only one node
EU_h=pd.DataFrame(columns=['start_node','end_node'])
EU['Mhte']=10**6  # maximum hydrogen transport cost
EU['Meth']=10**5  # maximum electricity transport cost
EU['feth']=0.7  # fraction of electricity in hydrogen
EU['fhte']=0.75  # fraction of electricity in hydrogen
EU['Mns'] = 100000
EU['Mnw'] = 10000
EU['Mnh'] = 10000000
EU_e['NTC']=1000  # maximum transportation cost for electricity
EU_h['MH']=500  # maximum transportation cost for hydrogen
costs = pd.DataFrame([["Italy",4000, 3000000, 10,2,200,1000,10000]],columns=["node","cs", "cw","ch","chte","ceth","cNTC","cMH"])

eu = Network()
eu.n = EU
eu.edgesP = EU_e
eu.edgesH = EU_h
eu.costs = costs
eu.genW_t = xr.DataArray(np.expand_dims(EW.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=EW.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.genS_t = xr.DataArray(np.expand_dims(ES.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=ES.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.loadH_t = xr.DataArray(np.expand_dims(HL.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=HL.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )
eu.loadE_t = xr.DataArray(np.expand_dims(EL.T.values, axis = 2), coords={'time': pd.date_range('2023-01-01 00:00:00', periods=EL.shape[1], freq='h'), 'node': ['Italy',], 'scenario': [1,]}, dims=['time', 'node', 'scenario'] )


#%% compute results

out_sing1 = OPT1(ES.to_numpy(),EW.to_numpy(),EL.to_numpy(),HL.to_numpy(),d=1,rounds=1)

#%%

out_sing2 = OPT2(eu,long_outs=True)


#%% results

"""

STARTING OPT1 -- setting up model for 1 batches of 1 scenarios.

OPT Model has been set up, this took  0.3647 s.
Round 1 of 1 - opt time: 1.035s.



STARTING OPT2 -- setting up model for 1 batches of 1 scenarios.

OPT Model has been set up, this took  2.4534 s.
Round 1 of 1 - opt time: 14.222s.



out_sing1
Out[118]: 
[[87328.0,
  451.0,
  3333390.141516082,
  14053.347524572586,
  1027.2159299583018,
  2082491174.8506062]]

out_sing2
Out[117]: 
[[array([87328.]),
  array([451.]),
  array([3333391.]),
  array([14054.]),
  array([1028.]),
  2082491174.8506062]]


Conclusion: on the same scenario, single node, the two methods give the same result, but the second takes much longer to compute.

"""


