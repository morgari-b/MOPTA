
#%%  VALIDATION FUNCTION

# ideas: 
#   - get average scenario, optimize on that
#   - loss function to optimal scenario
#   - rolling horizon optimization 2days at a time
#   - messages telling me when it fails

#TODO: capire perchè è feasible anche se non usa stoccaggio di idrogeno
#TODO: AAA sistemare relazione cona ggregazione temporale (così) i timestep della domanda sono sfasati con lo stoccaggio di idrogeno


def ValidateHfix(network,VARS,scenarios, day_initial, scenario_initial):
    """
    Like validate but starting from time t_initial and scenario_initial and doing one scenario at the time.
    """
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    H_agg=VARS[0]['H_dates'].copy()
    goalH = H_agg.mean(dim='scenario')
    init_date = pd.Timestamp(2023, 1, 1)
    T = network.T
    t = T
    # Create a new DataArray with the same values as goalH at time=0, but with the new time coordinate T
    new_data = goalH.sel(time=0).assign_coords(time=T)
    # Concatenate the new data along the time dimension with the original goalH
    goalH = xr.concat([goalH, new_data], dim="time")
    new_time = np.arange(0,T)
    goalH = goalH.interp(time=new_time)
    dates = [init_date + pd.Timedelta(hours=int(t)) for t in goalH.time.values]
    goalH.coords['date'] = ('time', dates)

    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    #d = network.n_scenarios 
    d = scenarios['wind_scenario'].shape[2]
    #inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    inst = 24
    #tp_obj = network.time_partition
        #tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    #ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    #nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    #nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    #mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    #meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    #addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    #addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)
    
    HtE = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0,obj=-0.05)
    P_edge_pos = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[i,k] <= nh[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( EtH[i,k] <= meth[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( HtE[i,k] <= mhte[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    
    # new variables for loss function
    loss = 1
    delta_H = model.addVars(product(range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
    if 'node' in network.n.columns:
        network.n = network.n.set_index('node')

    c1=model.addConstr(H[0,0]>=0)
    c2=model.addConstr(H[0,0]>=0)
    c3=model.addConstr(H[0,0]>=0)
    c4=model.addConstr(H[0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    T = len(pv_scenario.time)
    n_days = int(np.floor(T/24))
    H_list = []
    for j in range(scenario_initial,d):
        values = np.zeros([1,len(network.n.index.to_list())])
        if j == scenario_initial: #we start from day day_initial and scenario scenario_initial and then continue as normal
            day_num = day_initial
             # starting hydrogen level
            tp = network.time_partition.agg
            for i in range(len(tp)):
                if type(tp[i]) is list:
                    if day_num*24 in tp[i]:
                        day_num == tp[i][0] // 24
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=H_agg.sel(time=tp[i][0]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
                
                else:
                    if day_num == tp[i] // 24:
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=H_agg.sel(time=tp[i]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
            
        else:
            day_num = 0
             # starting hydrogen levels
            
            Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
            #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
            Hs[0,:]=H_agg.sel(time=0).max('scenario')+5*10**7 # else might get unfeasibility for net 0 solutions
        # start iterating
        for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
            if day_num < 365:    
                #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
                EW = wind_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
                ES = pv_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                HL = hydrogen_demand_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                EL = elec_load_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                
                #MAKE THE INDICES MAKE SENSE
                day_num=day_num+1
            
                model.remove(c1)
                model.remove(c2)
                model.remove(c3)
                model.remove(c4)
                #model.remove(c5)
                
                # if network.loadP_t_agg.shape[2] > 1:
                #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
            
                    
                #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
                
                #else:
            
                    
                # changed index compared to OPT3: H[i] is the storage at the end of hour i.
                c1 = model.addConstrs((- H[i,k] + H[i-1,k] + 30*network.n['feth'].iloc[k]*EtH[i,k] - HtE[i,k] -
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = i, node = k) for i in range(1,inst) for k in range(Nnodes)))
            
            
                c2 = model.addConstrs((ns[k]*float(ES.isel(time = i, node = k)) + nw[k]*float(EW.isel(time = i, node = k)) + 0.033*network.n['fhte'].iloc[k]*HtE[i,k] - EtH[i,k]
                                    - quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                                    + quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    >= EL.isel(time = i, node = k) for k in range(Nnodes)for i in range(inst)))
                
                # constrain to end of previous day
                #print([Hs.isel(time = -1,node = k) for k in range(Nnodes)])
                c3 = model.addConstrs((- H[0,k] + Hs.isel(time = -1,node = k) + 30*network.n['feth'].iloc[k]*EtH[0,k] - HtE[0,k] -
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = 0, node = k) for k in range(Nnodes)))

                
                # constraints for loss:
                #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                logging.debug(f"current day mean storage level {goalH.where(goalH.date.isin(pd.date_range(init_date,freq='h',periods=24)),drop=True).isel(node=0)}")
                c4 = model.addConstrs( delta_H[i,k] >= 5*10**7- H[i,k] + goalH.where(goalH.date.isin(pd.date_range(init_date,freq='h',periods=24)),drop=True).isel(node=k,time=i) for i in range(inst) for k in range(Nnodes))
                
                
                
                model.optimize()
                if model.Status!=2:
                    print("Status = {}".format(model.Status))
                    print(f"Failed at day {day_num}, scenario {j}")
                    return day_num, j
                else:
                    values = np.zeros([inst,len(network.n.index.to_list())])
                    for key in H:
                        #print(key)
                        values[key]=H[key].X
                    
                    Hss = xr.DataArray(values, 
                                        dims=["time","node"], 
                                        coords = dict(zip(["time","node"],[range(inst),network.n.index.to_list()])))
                    Hs = xr.concat([Hs,Hss],dim='time')
                    #logging.info("Day: {}, Scenario: {}, Hss: {}".format(day_num, j, Hss))
                    print('validation opt time: ',np.round(time.time()-start_time,4),'s. Day: ',day_num, "Scenario: ", j)

            else: #if we are in the last day
                break
    return day_num, j
#%% old Validate
def Validate(network,VARS,scenarios):
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    if 'node' in network.n.columns:
        network.n = network.n.set_index('node')
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    goalH=VARS[0]['H'].mean('scenario').to_pandas()
    t=goalH.shape[0]
    goalH.loc[t,:]=goalH.loc[0,:]
    goalH=goalH.set_index(pd.date_range("2023-01-01", "2024-01-01, 00:00:00",periods=t+1)).resample("h").interpolate("linear").head(-1)
    
    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    #d = network.n_scenarios 
    d = scenarios['wind_scenario'].shape[2]
    #inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    inst = 24
    #tp_obj = network.time_partition
        #tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    #ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    #nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    #nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    #mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    #meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    #addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    #addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
    # starting hydrogen levels
    values = np.zeros([d,1,len(network.n.index.to_list())])
    Hs = xr.DataArray(values, 
                         dims=["scenario","time","node"], 
                         coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range('01 jan,2023 00:00:00',freq='h',periods=1),network.n.index.to_list()])))
    #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
    Hs[:,0,:]=VARS[0]['H'][:,0,:].max('scenario') # else might get unfeasibility for net 0 solutions
    
    
    c1=model.addConstr(H[0,0,0]>=0)
    c2=model.addConstr(H[0,0,0]>=0)
    c3=model.addConstr(H[0,0,0]>=0)
    c4=model.addConstr(H[0,0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    
   
    day_num = 0
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
        #MAKE THE INDICES MAKE SENSE
        day_num=day_num+1
    
        model.remove(c1)
        model.remove(c2)
        model.remove(c3)
        model.remove(c4)
        #model.remove(c5)
        
        # if network.loadP_t_agg.shape[2] > 1:
        #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
    
            
        #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
        
        #else:
    
            
        # changed index compared to OPT3: H[i] is the storage at the end of hour i.
        c1 = model.addConstrs((- H[j,i,k] + H[j,i-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(1,inst) for k in range(Nnodes)))
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            break
        else:
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                #print(key)
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return Hs #day_num
def Validate3(network,VARS,scenarios):
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    goalH=VARS[0]['H'].mean('scenario').to_pandas()
    t=goalH.shape[0]
    goalH.loc[t,:]=goalH.loc[0,:]
    goalH=goalH.set_index(pd.date_range("2023-01-01", "2024-01-01, 00:00:00",periods=t+1)).resample("h").interpolate("linear").head(-1)
    
    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    #d = network.n_scenarios 
    d = scenarios['wind_scenario'].shape[2]
    #inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    inst = 24
    #tp_obj = network.time_partition
        #tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    #ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    #nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    #nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    #mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    #meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    #addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    #addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
    # starting hydrogen levels
    values = np.zeros([d,1,len(network.n.index.to_list())])
    Hs = xr.DataArray(values, 
                         dims=["scenario","time","node"], 
                         coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range('01 jan,2023 00:00:00',freq='h',periods=1),network.n.index.to_list()])))
    #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
    Hs[:,0,:]=VARS[0]['H'][:,0,:].max('scenario') # else might get unfeasibility for net 0 solutions
    
    
    c1=model.addConstr(H[0,0,0]>=0)
    c2=model.addConstr(H[0,0,0]>=0)
    c3=model.addConstr(H[0,0,0]>=0)
    c4=model.addConstr(H[0,0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    
   
    day_num = 0
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
        #MAKE THE INDICES MAKE SENSE
        day_num=day_num+1
    
        model.remove(c1)
        model.remove(c2)
        model.remove(c3)
        model.remove(c4)
        #model.remove(c5)
        
        # if network.loadP_t_agg.shape[2] > 1:
        #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
    
            
        #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
        
        #else:
    
            
        # changed index compared to OPT3: H[i] is the storage at the end of hour i.
        c1 = model.addConstrs((- H[j,i,k] + H[j,i-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(1,inst) for k in range(Nnodes)))
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            return day_num
        else:
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                #print(key)
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return day_num


def Validate2(network,VARS,scenarios, day_initial, scenario_initial):
    """
    Like validate but starting from time t_initial and scenario_initial and doing one scenario at the time.
    """
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    goalH=VARS[0]['H'].mean('scenario').to_pandas()
    t=goalH.shape[0]
    goalH.loc[t,:]=goalH.loc[0,:]
    goalH=goalH.set_index(pd.date_range("2023-01-01", "2024-01-01, 00:00:00",periods=t+1)).resample("h").interpolate("linear").head(-1)
    
    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    #d = network.n_scenarios 
    d = scenarios['wind_scenario'].shape[2]
    #inst = network.loadP_t_agg.shape[0] #number of time steps in time partition
    inst = 24
    #tp_obj = network.time_partition
        #tp = tp_obj.agg #time partition
    #print(f'sanity check, is inst equal to len tp= {inst == len(tp)}')
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    #model.setParam('Method',1)
    #time and scenario indipendent variables
    #ns = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cs,ub=network.n['Mns'])
    #nw = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=cw,ub=network.n['Mnw'])
    #nh = model.addVars(Nnodes,vtype=GRB.CONTINUOUS, obj=ch,ub=network.n['Mnh'])
    #mhte = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01, ub=network.n['Mhte'])
    #meth = model.addVars(Nnodes,vtype=GRB.CONTINUOUS,obj=0.01,ub=network.n['Meth'])
    #addNTC = model.addVars(NEedges,vtype=GRB.CONTINUOUS,obj=cNTC)
    #addMH = model.addVars(NHedges,vtype=GRB.CONTINUOUS,obj=cMH)
    
    HtE = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,lb=0) # expressed in kg
    EtH = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, lb=0) # expressed in MWh
    H = model.addVars(product(range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,lb=0)
    P_edge_pos = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[i,k] <= nh[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( EtH[i,k] <= meth[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( HtE[i,k] <= mhte[k] for i in range(inst) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    model.addConstrs( P_edge_pos[i,k] - P_edge_neg[i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for k in range(NEedges))
    model.addConstrs( H_edge_pos[i,k] - H_edge_neg[i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(inst),range(Nnodes)),obj = loss)#,lb=-GRB.INFINITY)
    
    if 'node' in network.n.columns:
        n.n = n.n.set_index('node')

    c1=model.addConstr(H[0,0]>=0)
    c2=model.addConstr(H[0,0]>=0)
    c3=model.addConstr(H[0,0]>=0)
    c4=model.addConstr(H[0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    T = len(pv_scenario.time)
    n_days = int(np.floor(T/24))
    H_list = []
    for j in range(scenario_initial,d):
        values = np.zeros([1,len(network.n.index.to_list())])
        if j == scenario_initial: #we start from day day_initial and scenario scenario_initial and then continue as normal
            day_num = day_initial
             # starting hydrogen level
            tp = network.time_partition.agg
            for i in range(len(tp)):
                if type(tp[i]) is list:
                    if day_num*24 in tp[i]:
                        day_num == tp[i][0] // 24
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=VARS[0]['H'].sel(time=tp[i][0]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
                
                else:
                    if day_num == tp[i] // 24:
                        Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
                        Hs[0,:]=VARS[0]['H'].sel(time=tp[i]).max('scenario') # else might get unfeasibility for net 0 solutions
                        break
            
        else:
            day_num = 0
             # starting hydrogen levels
            
            Hs = xr.DataArray(values, 
                                dims=["time","node"], 
                                coords = dict(zip(["time","node"],[range(1),network.n.index.to_list()])))
            #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
            Hs[0,:]=VARS[0]['H'].sel(time=0).max('scenario') # else might get unfeasibility for net 0 solutions
        # start iterating
        for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
            if day_num < 365:    
                #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
                EW = wind_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
                ES = pv_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = j)
                HL = hydrogen_demand_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                EL = elec_load_scenario.isel(time = slice(24*day_num,24*(day_num+1)), scenario = 0)
                
                #MAKE THE INDICES MAKE SENSE
                day_num=day_num+1
            
                model.remove(c1)
                model.remove(c2)
                model.remove(c3)
                model.remove(c4)
                #model.remove(c5)
                
                # if network.loadP_t_agg.shape[2] > 1:
                #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
            
                    
                #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
                #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
                
                #else:
            
                    
                # changed index compared to OPT3: H[i] is the storage at the end of hour i.
                c1 = model.addConstrs((- H[i,k] + H[i-1,k] + 30*network.n['feth'].iloc[k]*EtH[i,k] - HtE[i,k] -
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[i,l]-H_edge_neg[i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = i, node = k) for i in range(1,inst) for k in range(Nnodes)))
            
            
                c2 = model.addConstrs((ns[k]*float(ES.isel(time = i, node = k)) + nw[k]*float(EW.isel(time = i, node = k)) + 0.033*network.n['fhte'].iloc[k]*HtE[i,k] - EtH[i,k]
                                    - quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                                    + quicksum(P_edge_pos[i,l] - P_edge_neg[i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                                    >= EL.isel(time = i, node = k) for k in range(Nnodes)for i in range(inst)))
                
                # constrain to end of previous day
                #print([Hs.isel(time = -1,node = k) for k in range(Nnodes)])
                c3 = model.addConstrs((- H[0,k] + Hs.isel(time = -1,node = k) + 30*network.n['feth'].iloc[k]*EtH[0,k] - HtE[0,k] -
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                                quicksum(H_edge_pos[0,l]-H_edge_neg[0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                                == HL.isel(time = 0, node = k) for k in range(Nnodes)))

                
                # constraints for loss:
                #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
                c4 = model.addConstrs( delta_H[i,k] >= - H[i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for i in range(inst) for k in range(Nnodes))
                
                
                
                model.optimize()
                if model.Status!=2:
                    print("Status = {}".format(model.Status))
                    print(f"Failed at day {day_num}, scenario {j}")
                    return day_num, j
                else:
                    values = np.zeros([inst,len(network.n.index.to_list())])
                    for key in H:
                        #print(key)
                        values[key]=H[key].X
                    
                    Hss = xr.DataArray(values, 
                                        dims=["time","node"], 
                                        coords = dict(zip(["time","node"],[range(inst),network.n.index.to_list()])))
                    Hs = xr.concat([Hs,Hss],dim='time')
                    print('validation opt time: ',np.round(time.time()-start_time,4),'s. Day: ',day_num, "Scenario: ", j)

            else: #if we are in the last day
                break
    return day_num, j

#%%
def Validate_mhte(network,VARS,scenarios,free_mhte=True):
         
    ns = VARS[0]["ns"]
    nw = VARS[0]["nw"]
    nh = VARS[0]["nh"]
    mhte = VARS[0]["mhte"] 
    meth = VARS[0]["meth"]
    addNTC = VARS[0]["addNTC"] 
    addMH = VARS[0]["addMH"]
    
    
    # average hydrogen storage levels in train scenarios, set as goal for loss function
    goalH=VARS[0]['H'].mean('scenario').to_pandas()
    t=goalH.shape[0]
    goalH.loc[t,:]=goalH.loc[0,:]
    goalH=goalH.set_index(pd.date_range("2023-01-01", "2024-01-01, 00:00:00",periods=t+1)).resample("h").interpolate("linear").head(-1)
    
    # costs    
    if network.costs.shape[0] == 1: #if the costs are the same:
       cs, cw, ch, ch_t, chte, ceth, cNTC, cMH, cH_edge, cP_edge = network.costs['cs'][0], network.costs['cw'][0], network.costs['ch'][0], network.costs['ch_t'][0], network.costs['chte'][0], network.costs['ceth'][0], network.costs['cNTC'][0], network.costs['cMH'][0], network.costs['cH_edge'][0], network.costs['cP_edge'][0]
    else:
        print("add else") #actually we can define the costs appropriately using the network class directly
    
    # set up model for 24h
    start_time=time.time()
    Nnodes = network.n.shape[0]
    NEedges = network.edgesP.shape[0]
    NHedges = network.edgesH.shape[0]
    d = scenarios['wind_scenario'].shape[2]
    inst = 24
    
    env = Env(params={'OutputFlag': 0})
    model = Model(env=env)
    model.setParam('LPWarmStart',1)
    
    HtE = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=chte/d,ub=mhte) # expressed in kg
    EtH = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS, obj=ceth/d, ub=meth) # expressed in MWh
    H = model.addVars(product(range(d),range(inst),range(Nnodes)),vtype=GRB.CONTINUOUS,ub=nh)
    P_edge_pos = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    P_edge_neg = model.addVars(product(range(d),range(inst),range(NEedges)),vtype=GRB.CONTINUOUS, obj=cP_edge, lb=0)
    H_edge_pos = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    H_edge_neg = model.addVars(product(range(d),range(inst),range(NHedges)),vtype=GRB.CONTINUOUS, obj=cH_edge, lb=0)
    
    #todo: add starting capacity for generators (the same as for liners)
    model.addConstrs( H[j,i,k] <= nh[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    # model.addConstrs( EtH[j,i,k] <= meth[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    # model.addConstrs( HtE[j,i,k] <= mhte[k] for i in range(inst) for j in range(d) for k in range(Nnodes))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] <= (network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] <= (network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    model.addConstrs( P_edge_pos[j,i,k] - P_edge_neg[j,i,k] >= -(network.edgesP['NTC'].iloc[k] + addNTC[k]) for i in range(inst) for j in range(d) for k in range(NEedges))
    model.addConstrs( H_edge_pos[j,i,k] - H_edge_neg[j,i,k] >= -(network.edgesH['MH'].iloc[k] + addMH[k]) for i in range(inst) for j in range(d) for k in range(NHedges))
    
    # new variables for loss function
    loss = 0.1
    delta_H = model.addVars(product(range(d),range(inst),range(Nnodes)),obj = loss,lb=-GRB.INFINITY)
    
    # starting hydrogen levels
    values = np.zeros([d,1,len(network.n.index.to_list())])
    Hs = xr.DataArray(values, 
                         dims=["scenario","time","node"], 
                         coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range('01 jan,2023 00:00:00',freq='h',periods=1),network.n.index.to_list()])))
    #Hs[:,0,:]=goalH.loc['jan 1 23'].iloc[0,:]
    Hs[:,0,:]=VARS[0]['H'][:,0,:].max('scenario') # else might get unfeasibility for net 0 solutions
    
    
    c1=model.addConstr(H[0,0,0]>=0)
    c2=model.addConstr(H[0,0,0]>=0)
    c3=model.addConstr(H[0,0,0]>=0)
    c4=model.addConstr(H[0,0,0]>=0)
    #c5=model.addConstr(H[0,0,0]>=0)
    
    wind_scenario = scenarios['wind_scenario']
    pv_scenario = scenarios['pv_scenario']
    hydrogen_demand_scenario = scenarios['hydrogen_demand_scenario']
    elec_load_scenario = scenarios['elec_load_scenario']
    
   
    day_num = 0
    MHTE=mhte
    METH=meth
    # start iterating
    for day in pd.date_range('Jan 01 2023','Dec 31 2023',freq='d'):
        

        #EW = wind_scenario.sel(time=[ str(each) for each in pd.date_range('2024'+str(day)[4:],periods=24,freq='h').to_list()])
        EW = wind_scenario[24*day_num:24*(day_num+1),:,:]
        #ES = pv_scenario.sel(time=[ str(each) for each in pd.date_range(day,periods=24,freq='h').to_list()])
        ES = pv_scenario[24*day_num:24*(day_num+1),:,:]
        HL = hydrogen_demand_scenario[24*day_num:24*(day_num+1),:,:]
        EL = elec_load_scenario[24*day_num:24*(day_num+1),:,:]
        
        #MAKE THE INDICES MAKE SENSE
        day_num=day_num+1
    
        model.remove(c1)
        model.remove(c2)
        model.remove(c3)
        model.remove(c4)
        #model.remove(c5)
        
        # if network.loadP_t_agg.shape[2] > 1:
        #     c1 = model.addConstrs((- H[j,(i+1)%inst,k] + H[j,i,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                     quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                     == HL[i,k,j] for j in range(d) for i in range(inst) for k in range(Nnodes)))
    
            
        #     c2 = model.addConstrs((ns[k]*ES[i,k,j] + nw[k]*EW[i,k,j] + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k] -
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list()) +
        #                         quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
        #                         >= EL[i,k,j] for k in range(Nnodes) for j in range(d) for i in range(inst)))
        
        #else:
    
            
        # changed index compared to OPT3: H[i] is the storage at the end of hour i.
        c1 = model.addConstrs((- H[j,i,k] + H[j,i-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,i,k] - HtE[j,i,k] -
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,i,l]-H_edge_neg[j,i,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[i,k,0] for j in range(d) for i in range(1,inst) for k in range(Nnodes)))
    
    
        c2 = model.addConstrs(ns[k]*float(ES[i,k,j]) + nw[k]*float(EW[i,k,j]) + 0.033*network.n['fhte'].iloc[k]*HtE[j,i,k] - EtH[j,i,k]
                             - quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['start_node']==network.n.index.to_list()[k]].index.to_list())
                            + quicksum(P_edge_pos[j,i,l] - P_edge_neg[j,i,l] for l in network.edgesP.loc[network.edgesP['end_node']==network.n.index.to_list()[k]].index.to_list())
                            >= EL[i,k,0] for k in range(Nnodes) for j in range(d) for i in range(inst))
        
        # constrain to end of previous day
        c3 = model.addConstrs((- H[j,0,k] + Hs[j,-1,k] + 30*network.n['feth'].iloc[k]*EtH[j,0,k] - HtE[j,0,k] -
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['start_node']==network.n.index.to_list()[k]].index.to_list()) +
                        quicksum(H_edge_pos[j,0,l]-H_edge_neg[j,0,l] for l in network.edgesH.loc[network.edgesH['end_node']==network.n.index.to_list()[k]].index.to_list())
                        == HL[0,k,0] for j in range(d) for k in range(Nnodes)))
    
    
        
        # constraints for loss:
        #c4 = model.addConstrs( H[j,i,k] - delta_H[j,i,k] <=  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        #c4 = model.addConstrs( H[j,i,k] + delta_H[j,i,k] ==  goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        c4 = model.addConstrs( delta_H[j,i,k] >= - H[j,i,k] + goalH.loc[pd.date_range(day,freq='h',periods=inst)].iloc[i,k] for j in range(d) for i in range(inst) for k in range(Nnodes))
        
        
        
        model.optimize()
        if model.Status!=2:
            print("Status = {}".format(model.Status))
            print("Failed at " + str(day))
            break
        else:
            
            max_mhte=max([max([max([HtE[j,i,k].X for i in range(inst)]) for j in range(d)]) for k in range(Nnodes)])
            max_meth=max([max([max([EtH[j,i,k].X for i in range(inst)]) for j in range(d)]) for k in range(Nnodes)])
            if MHTE<max_mhte:
                MHTE=max_mhte
            if METH<max_meth:
                METH=max_meth
            values = np.zeros([d,inst,len(network.n.index.to_list())])
            for key in H:
                values[key]=H[key].X
            
            Hss = xr.DataArray(values, 
                                 dims=["scenario","time","node"], 
                                 coords = dict(zip(["scenario","time","node"],[range(d),pd.date_range(day,freq='h',periods=inst),network.n.index.to_list()])))
            Hs = xr.concat([Hs,Hss],dim='time')
            print('opt time: ',np.round(time.time()-start_time,4),'s. Day: ',str(day))
    return {'day_num':day_num, 'Hs':Hs, 'MHTE':MHTE,'METH':METH}


def import_scenario_val(start,stop):
    
    path = "model/scenario_generation/scenarios/"
    elec_load_df = pd.read_csv(path+'electricity_load_2023.csv')
    elec_load_df = elec_load_df[['DateUTC', 'IT', 'ES', 'AT', 'FR','DE']]
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[['IT', 'ES', 'AT', 'FR','DE']].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node': ['Italy', 'Spain', 'Austria', 'France','Germany'], 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    
    ave = [31532.209018, 26177.184589, 6645.657078, 48598.654281, 52280.658229 ]
    a = xr.DataArray(ave,dims=['node'], coords={'node':['Italy','Spain','Austria','France','Germany']})
    elec_load_scenario=elec_load_scenario*a
    
    #wind_scenario = 4*scenario_to_array(pd.read_csv(path +'small-eu-wind-scenarios3.csv', index_col = 0))
    wind_scenario = 4*import_scenario(path + 'small-eu-wind-scenarios3.csv')
    pv_scenario = 0.01*import_scenario(path + 'small-eu-PV-scenarios.csv')
    
    
    df = pd.read_csv(path+'hydrogen_demandg.csv', index_col = 0).head()
    time_index = range(df.shape[1])#pd.date_range('2023-01-01 00:00:00', periods=df.shape[1], freq='H')
    node_names=['Italy', 'Spain', 'Austria', 'France','Germany']

    hydro = xr.DataArray(
        np.expand_dims(df.T.values, axis = 2),
        coords={'time': time_index, 'node': node_names, 'scenario': [0]},
        dims=['time', 'node', 'scenario'] )
    hydro_mean = hydro.mean(dim = ["time","scenario"]) 
    hydrogen_demand_scenario = hydro / hydro_mean
    
    
    #hydrogen_demand_scenario2 = import_scenario(path + 'hydrogen_demandg.csv')
    
    #eu.add_scenarios(wind_scenario * max_wind, pv_scenario * max_solar, hydrogen_demand_scenario, elec_load_scenario)
    
    scenarios = {
        'wind_scenario' : wind_scenario.sel(scenario=slice(start,stop)),
        'pv_scenario' : pv_scenario.sel(scenario=slice(start,stop)),
        'hydrogen_demand_scenario' : hydrogen_demand_scenario,
        'elec_load_scenario' : elec_load_scenario
        }
    
    
    return scenarios

# %%