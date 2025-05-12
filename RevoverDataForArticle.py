
#%%
# import dash
import os
os.chdir("C:/Users/ghjub/codes/MOPTA")
import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
# Import the necessary functions and classes
#from scenario_generation.scenario_generation import read_parameters, SG_beta, SG_weib, quarters_df_to_year
#from model.prova_bianca import OPT
from model.YUPPY import Network
from model.OPT_methods import OPT_time_partition, OPT3, OPT_agg
from model.EU_net import EU
from model.scenario_generation.scenario_generation import import_generated_scenario, generate_independent_scenarios, import_scenarios, plot_scenarios_df, scenario_to_array, country_name_to_iso
from model.OPloTs import plotOPT3_secondstage, plotOPT_time_partition, node_results_df, plotOPT_pie, plotE_balance, plotH_balance
#%%
scenarios_path = "model/scenario_generation/scenarios/"
# %% define network
network = EU()
#%% import scenarios
network.n.reset_index(inplace=True)
scenarios = import_scenarios("small-EU")
plots = {}
samples = 10
plots["wind"] = scenarios["wind"][scenarios["wind"]["scenario"] < samples]
plots["PV"] = scenarios["PV"][scenarios["PV"]["scenario"] < samples]
scenarios["wind"] = scenarios["wind"][scenarios["wind"]["scenario"] < samples]
scenarios["PV"] = scenarios["PV"][scenarios["PV"]["scenario"] < samples]

#print(type(scenarios["wind"]))
network.genW_t = scenario_to_array(scenarios["wind"])
network.genS_t = scenario_to_array(scenarios["PV"])
wind_figs = plot_scenarios_df(plots["wind"], var_name = "p.u. Wind power output", title1 = "Wind power output in each node", title2 = "Wind power output for various scenarios" )
pv_figs = plot_scenarios_df(plots["PV"], var_name = "p.u. PV power output", title1 = "PV power output in each node", title2 = "PV power output for various scenarios")

#%%
locations = network.n.location.unique().tolist()
scen_path = 'model/scenario_generation/scenarios/'
elec_load_df = pd.read_csv(scen_path+'electricity_load_2023.csv')
time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')
scenario = 0
elec_load_scenario = xr.DataArray(
    np.expand_dims(elec_load_df[[country_name_to_iso(location) for location in locations]].values, axis = 2), #add one dimension to correspond with scenarios
    coords={'time': time_index, 'node':locations, 'scenario': [0]},
    dims=['time', 'node', 'scenario']
)
#print(network.n)
hydrogen_demand_scenario = import_generated_scenario(scen_path+'hydrogen_demandg.csv',len(locations), scenario, node_names=locations)

# %% Duck Calculations

total_demand = elec_load_scenario.sel(node = 'Italy',scenario=0).sum()
total_pv = network.genS_t.sel(node = 'Italy',scenario = 0).sum()

perc = 1 / 2
demand = elec_load_scenario.sel(node = 'Italy',scenario=0).copy()
solar = network.genS_t.sel(node = 'Italy',scenario = 0).copy() * total_demand / total_pv * perc
time = pd.to_datetime(solar.time.values)
demand.coords['time'] = time
solar.coords['time'] = time
net = demand - solar

# %% plto duck
def daily_mean(ds):
    # Group by day and calculate the mean
    daily_mean = ds.groupby("time.date").mean()

    return daily_mean

x = solar.time
y_solar = solar.values
y_demand = elec_load_scenario.sel(node = 'Italy',scenario=0).values
y_net = net.values


x2 = daily_mean(solar).date
y2_solar = daily_mean(solar).values
y2_demand = daily_mean(demand).rolling(date=14, center=True).mean().values
y2_net = daily_mean(net).rolling(date=14, center=True).mean().values
# Create figure
fig = go.Figure()

# Add a trace (line plot)
fig.add_trace(go.Scatter(x=x, y=y_solar, mode='lines', name='Solar'))

# Add a trace (line plot)
fig.add_trace(go.Scatter(x=x, y=y_demand, mode='lines', name='Energy Demand'))

# Add a trace (line plot)
fig.add_trace(go.Scatter(x=x, y=y_net, mode='lines', name='Net Energy'))

# Show figure
fig.show()

fig2 = go.Figure()

# Add a trace (line plot)
fig2.add_trace(go.Scatter(x=x2, y=y2_solar, mode='lines', name='Solar'))

# Add a trace (line plot)
fig2.add_trace(go.Scatter(x=x2, y=y2_demand, mode='lines', name='Energy Demand'))

# Add a trace (line plot)
fig2.add_trace(go.Scatter(x=x2, y=y2_net, mode='lines', name='Net Energy'))

# Show figure
fig2.show()


# %%
def save_for_tex(filename, data):
    """
    Save a NumPy array or Pandas Series/DataFrame in a PGFPlots-compatible format.
    Args:
    - filename: str, name of the output file.
    - data: NumPy array or Pandas DataFrame/Series.
    """
    # Convert to DataFrame if it's not already
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame()

    # Save with space as delimiter, floating point precision for floats
    data.to_csv(filename, sep=' ', index=True, header=False, float_format='%.6f')

    print(f"File saved: {filename}")

#%%
dates = dates = pd.date_range(start='2024-01-01', periods=len(y2_solar), freq='D')  # One week of hourly data
#np.savetxt('x2.csv', np.column_stack((x2, x2)), delimiter=' ')
# %%
save_for_tex('y2_solar.csv', y2_solar)
save_for_tex('y2_demand.csv', y2_demand)
save_for_tex('y2_net.csv', y2_net)
# %%

def daily_hourly_average(v):
    """
    Compute the hourly average for each hour across all days of a year.
    
    Parameters:
    v (numpy.Sndarray): 8760-length array with hourly data for a year (365 days * 24 hours).
    
    Returns:
    numpy.ndarray: 24-length array with the average for each hour of the day.
    """
    if len(v) != 8760:
        raise ValueError("Input array must have a length of 8760 (365 days * 24 hours).")
    
    # Reshape into 365 days × 24 hours and compute the mean along the days
    v_reshaped = v.reshape(365, 24)
    hourly_avg = v_reshaped.mean(axis=0)
    
    return hourly_avg

# Example usage:

# %%
y_net_hour = daily_hourly_average(y_net)
y_solar_hour = daily_hourly_average(y_solar)
y_demand_hour = daily_hourly_average(y_demand)
save_for_tex('yh_solar.csv', y_solar_hour)
save_for_tex('yh_demand.csv', demand.groupby('time.hour').mean(dim='time').values)
save_for_tex('yh_net.csv', y_net_hour)
# %%s
