
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:09:24 2024

@author: frulcino

This scripts contains functions to model multivariate variables by:
    1) Fitting indipendently their marginal distributions
    2) Coupling the variables by fitting with a Gaussian Copula
"""
#%% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.cm as cm
from reliability.Fitters import Fit_Weibull_2P
import os
from scipy.sparse import dia_matrix, save_npz, load_npz
#os.chdir('/home/frulcino/codes/MOPTA/')


#TODOS
# - Can I use Y instead of O_h directly?
# %% helper functions

def read_parameters(country):
    """
    Get parameters as saved in main()
    """
    params = {}
    wind_params_df = pd.read_csv(f"parameters/marginals-wind-{country}", header = [0,1], index_col= 0)
    wind_params_df.columns = pd.MultiIndex.from_tuples([(int(x[0]), int(x[1])) for x in wind_params_df.columns], names = ["day","hour"])
    PV_params_df = pd.read_csv(f"parameters/marginals-PV-{country}", header = [0,1], index_col = 0)
    PV_params_df.columns = pd.MultiIndex.from_tuples([(int(x[0]), int(x[1])) for x in PV_params_df.columns], names = ["day","hour"])
    night_hours = pd.read_csv(f"parameters/night_hours-PV-{country}", header = 0, index_col = 0)
    night_hours = list(zip(list(night_hours.iloc[:,0]),list(night_hours.iloc[:,1])))
    cov_wind = load_npz(f"parameters/copula-wind-{country}.npz").toarray()
    cov_PV = load_npz(f"parameters/copula-PV-{country}.npz").toarray()
    params["wind"] = wind_params_df, cov_wind
    params["PV"] = PV_params_df, cov_PV, night_hours
    return params
 
def date_format(df, country):
    grouped = df[country].groupby(lambda date: (date.dayofyear, date.hour), axis = 0)
    ran = False
    n_instances = len(grouped)
    X = {}
    for time_instance, time_instance_df in grouped:
        time_instance_df.index = time_instance_df.index.year
        X[time_instance] = time_instance_df          
    X = pd.DataFrame(X).sort_index()
    X.columns.names = "day", "hour"
    
    #drop last day:
    X = X.loc[:,:(365,23)]
    return X
      #print(time_instance, time_instance_df)
      #  

# %% generate multivariate data decomposition

def sample_multivariate_normal(L, mean, num_samples):
    """
    Generate samples from a multivariate normal distribution, given a lower triangular matrix decomposition 

    Parameters
    ----------
    L : array_like
        Lower triangular matrix such that L @ L.T = Covariance matrix
    mean : array_like
        Mean vector
    num_samples : int
        Number of samples

    Returns
    -------
    samples : array_like
        Samples from the multivariate normal distribution
    """
    num_dim = L.shape[0]
    z = np.random.randn(num_samples, num_dim)
    samples = z @ L + mean
    return samples


# %% wind fit
def fit_weibull(Y, method = "LS"):
    """
    Y : pandas dataframe having as columns variables and as rows scenarios
    returns dataframe having estimated with MLE weibull parameters.
    """
    parameters_df = pd.DataFrame(columns = Y.columns)
    for var in Y.columns:
        #params = stats.exponweib.fit(Y[var], method = "MLE")
        params = Fit_Weibull_2P(failures=Y[var].values, print_results = False, show_probability_plot = False, method = method)
        parameters_df.loc["scale", var],parameters_df.loc["shape", var] = params.alpha, params.beta
        
    return parameters_df


    
def fit_multivariate_weib(Y, simplify = False, max_dist = 24*3, method = "MLE"):
    """
    Fit data with multivariate distribution having Weibull marginals coupled with Gaussian copula.
    input
        Y: Pandas dataframe taking having as columns variables and rows as samples
    output
        parameters_df: df having for each variable the estimated parameters of the Weibull distribution
        E_h: the estimated covariance matrix of the Gaussian Copula
        
    """
    #weibul distribution estimation
    parameters_df = fit_weibull(Y, method)

    #copula estimation
    
    X_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    O_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    for var in Y.columns:
        #bring data to uniform domain
        X_h[var] = stats.exponweib.cdf(Y[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        #O_h[var] = stats.exponweib.ppf(X_h[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        O_h[var] = stats.norm.ppf(X_h[var]) 
        
    #copula covariance estimation:
    E_h = np.cov(O_h.T, ddof = 1)
    
    
    c= plt.imshow(E_h, interpolation = "nearest")
    plt.colorbar(c)
    plt.title("Gaussian copula covariance matrix")
    
    if simplify:
        E_shape = E_h.shape[0]
        row_index, col_index =np.indices(E_h.shape)
        distance_matrix = np.abs(row_index - col_index)
        mask = distance_matrix > max_dist
        E_h[mask] = 0
        
    
    return parameters_df, E_h

def SG_weib(n_scenarios, parameters_df, E_h, precomputed_decomposition = False):
    """

    Parameters
    ----------
    n_scenarios : integer
        number of scenarios to generate
    parameters_df : panda DataFrame
        having as columns the variables and as rows "scale" and "shape" parameters of 
        the Weibull marginal distributions
    E_h : np.array n_var x n_var
        estimated covariance matrix of the Gaussian copula if precomputed_decomposition = False
        or the cholesky decomposition L of the covariance matrix if precomputed_decomposition = True
    precomputed_decomposition : boolean
        if true, the cholesky decomposition of the covariance matrix is precomputed

    Returns
    -------
    scenarios: panda DataFrame
        each row is a scenario

    """
    n_vars = len(parameters_df.columns)
    var_names = pd.date_range("2024/01/01", periods = n_vars, freq = "h") #better names
    parameters_df.columns = var_names

    if precomputed_decomposition:
        L = E_h
        X_scen = pd.DataFrame(sample_multivariate_normal(L, np.zeros(n_vars), n_scenarios), columns = var_names)
    else:
        X_scen = pd.DataFrame(np.random.multivariate_normal(np.array([0]*n_vars), E_h, n_scenarios), columns = var_names)
   


    print("generated random uniform")
    #convert to uniform domain U and the to inidital domain Y
    U_scen = pd.DataFrame( columns=var_names)
    scenarios = pd.DataFrame( columns=var_names) #scenario dataframe

    for var in var_names:
        U_scen[var] = stats.norm.cdf(X_scen[var])
        scenarios[var] = stats.exponweib.ppf(U_scen[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        
    return scenarios

def SG_weib_example(n_samples = 100, n_scenarios = 100, n_vars = 24, shape = 2, scale = 3, bins = 20):
    """
    Example with randomly generated indipendent weibull distibutions
    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 100.
    n_scenarios : TYPE, optional
        DESCRIPTION. The default is 100.
    n_vars : TYPE, optional
        DESCRIPTION. The default is 24.
    shape : TYPE, optional
        DESCRIPTION. The default is 2.
    scale : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None.

    """
    #generate random scenarios with weibull distribution
    time_range = pd.date_range(start="20210101", end="20210102", periods=n_vars)
    Y = pd.DataFrame(scale*np.random.weibull(shape, (n_samples,n_vars)), columns = time_range)

    fig, axes = plt.subplots(2,3, figsize = (15,10))
    fig.suptitle("Overview")

    axes[0,0].hist(Y.iloc[:,0], bins = bins)
    axes[0,0].set_title("Data Histogram")

    parameters_df, E_h = fit_multivariate_weib(Y)
    
    ax = axes[0,1]
    c = ax.imshow(E_h, interpolation = "nearest")
    fig.colorbar(c, ax=ax)
    ax.set_title("Gaussian copula covariance matrix")
    
       
    # Plotting estimated parameters from a DataFrame
    ax = axes[0,2]
    
    # Plotting estimated parameters from a DataFrame
    ax.plot(parameters_df.loc["scale"], label='Estimated Scale', color='blue', linestyle='--')
    ax.plot(parameters_df.loc["shape"], label='Estimated Shape', color='green', linestyle='--')
    
    # Plotting actual parameters using horizontal lines
    ax.hlines(scale, xmin = Y.columns[0], xmax = Y.columns[-1], colors='blue', linestyles='-', label='Actual Scale')
    ax.hlines(shape, xmin = Y.columns[0], xmax = Y.columns[-1], colors='green', linestyles='-', label='Actual Shape')
    
    # Adding labels
    ax.set_xlabel('Index')  # Adjust the label as necessary depending on the index meaning
    ax.set_ylabel('Parameter Value')
    ax.set_title('Comparison of Estimated and Actual Parameters')

    # Add a legend to differentiate between plots
    ax.legend()

    scenarios = SG_weib(n_scenarios,Y.columns, parameters_df, E_h)



    axes[1,0].hist(scenarios.iloc[:,0], bins = bins)
    axes[1,0].set_title("Data Scenarios")

    plt.show()
  
# %% solar fit
def betapar_alpha(mean, var):
    return mean*((mean*(1-mean)/var)-1)

def betapar_beta(mean,var):
    return (1-mean)*((mean*(1-mean)/var)-1)

def fit_beta(Y):
    """
    Y : pandas dataframe having as columns variables and as rows scenarios
    returns dataframe having estimated with Moments estimator, look at stackexchange for explanation
    """
    
    night_hours = list(Y.iloc[:,list(Y.mean() <= 0.05)].columns)
    #print(night_hours)
    Y = Y.iloc[:,list(Y.mean() > 0.05)]
    parameters_df = pd.DataFrame(columns = Y.columns)
    
    for var in Y.columns:
        Y_var = Y[var]
        Y_mean = Y_var.mean()
        Y_var = Y_var.var()
        #params = stats.beta.fit(Y_var[Y_var >= 0.01], floc = 0, fscale = 1)
        parameters_df.loc["zero_weight", var] = 0 #probability of solar production to be zero
        alpha, beta =  betapar_alpha(Y_mean, Y_var), betapar_beta(Y_mean, Y_var)
        parameters_df.loc["alpha", var],parameters_df.loc["beta", var] = alpha, beta
        
    return parameters_df, night_hours

def fit_multivariate_beta(Y, simplify = False, max_dist = 24*3):
    """
    Fit data with multivariate distribution having Weibull marginals coupled with Gaussian copula.
    input
        Y: Pandas dataframe taking having as columns variables and rows as samples
        mask: if True, set cov = 0 for variables which are max_dist apart
        max_dist: distance between variables after which we assume independence
    output
        parameters_df: df having for each variable the estimated parameters of the Weibull distribution
        E_h: the estimated covariance matrix of the Gaussian Copula
        
    """
    #def sbeta_cdf(zero_weight, alpha, beta):
    #    """
    #    beta cdf with a non zero weight at zero
    #    """
    #weibul distribution estimation
    parameters_df, night_hours = fit_beta(Y)

    Y = Y.iloc[:,list(Y.mean() > 0.05)]
    X_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    O_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    for var in Y.columns:
        #bring data to uniform domain
        alpha, beta = parameters_df.loc["alpha", var], parameters_df.loc["beta", var]
        X_h[var] = stats.beta.cdf(Y[var], alpha, beta)
        # O_h[var] = stats.exponweib.ppf(X_h[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        O_h[var] = stats.norm.ppf(X_h[var]) 
        
    #copula covariance estimation:
    E_h = np.cov(O_h.T, ddof = 1) #non posso usare Y_h direttamente?
    
    c= plt.imshow(E_h, interpolation = "nearest")
    plt.colorbar(c)
    plt.title("Gaussian copula covariance matrix")
    if simplify:
        E_shape = E_h.shape[0]
        row_index, col_index =np.indices(E_h.shape)
        distance_matrix = np.abs(row_index - col_index)
        mask = distance_matrix > max_dist
        E_h[mask] = 0
        
    return parameters_df, E_h, night_hours
#%%
def SG_beta(n_scenarios, parameters_df, E_h, night_hours, save = True, filename = "PV_scenario"):
    """
    Parameters
    ----------
    n_scenarios : integer
    number of scenarios to generate
    parameters_df : panda DataFrame
    having as columns the variables and as rows "scale" and "shape" parameters of 
    the Weibull marginal distributions
    E_h : np.array n_var x n_var
    estimated covariance matrix of the Gaussian copula
    
    Returns
    -------
    scenarios: panda DataFrame
    each row is a scenario
    
    """
    var_names = parameters_df.columns #better names
    n_vars = len(var_names)
    X_scen = pd.DataFrame(np.random.multivariate_normal(np.array([0]*n_vars), E_h, n_scenarios), columns = var_names)
    #convert to uniform domain U and the to initial domain Y
    U_scen = pd.DataFrame( columns=var_names)
    scenarios = pd.DataFrame( columns=var_names) #scenario dataframe
    night_scenarios = pd.DataFrame(np.zeros((n_scenarios, len(night_hours))), columns=night_hours)
    for var in var_names:
        U_scen[var] = stats.norm.cdf(X_scen[var])
        alpha, beta = parameters_df.loc["alpha",var], parameters_df.loc["beta",var]
        scenarios[var] = stats.beta.ppf(U_scen[var], alpha, beta)
    
    scenarios = pd.concat([scenarios, night_scenarios], axis = 1)
    scenarios = scenarios.reindex(sorted(scenarios.columns), axis = 1)
    n_hours = len(scenarios.columns)
    scenarios.columns = pd.date_range("01/01/2023", periods = n_hours, freq="h")
    if save:
        scenarios.to_csv("scenarios/"+filename+f"{n_scenarios}")
        
       
    return scenarios
    
    
def quarters_df_to_year(file_name, location, df, column, location_column, save = True, n_scenarios = 100, n_hours = 24*365):
    """
    Convert challenge dataset, to yearly dataset
    """
    demand_scenarios = pd.DataFrame(np.zeros((n_scenarios,n_hours)), columns = pd.date_range("01/01/2023", periods = n_hours, freq = "h"))
    m_to_s = [f"Q{month%12// 3 + 1}" for month in range(1, 13)] #month to season
    for month in np.arange(12):
        season = m_to_s[month]
        season_df = df.loc[df["Quarter"] == season] #fetch season
        season_df = season_df.loc[season_df[location_column] == location] #fetch location
        season_df = season_df[column].groupby(lambda x: x//4).sum()
        S_demand =  demand_scenarios.iloc[:,(demand_scenarios.columns.month -1== month)]
        n_rows, n_cols = S_demand.shape
        demand_scenarios.iloc[:,(demand_scenarios.columns.month -1== month)] = np.array([list(season_df)* (n_cols // 24)]*n_scenarios)
    if save:
        demand_scenarios.to_csv("scenarios/"+file_name+f"-{location}.csv")
    return demand_scenarios
 #%% fetch data
iso_to_country = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CY': 'Cyprus',
        'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
        'DE': 'Germany', 'GR': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland, Republic of (EIRE)',
        'IT': 'Italy', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta',
        'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia',
        'SI': 'Slovenia', 'ES': 'Spain', 'SE': 'Sweden', 'GB': 'United Kingdom', 'AL': 'Albania',
        'BA': 'Boznia and Herzegovina', 'CH': 'Switzerland',  'MK': 'North Macedonia', 'RS': 'Serbia',
        'MD': 'Moldavia', 'SK': 'Slovakia', 'ME': 'Montenegro', 'NO': 'Norway'
    }
# %%
def fit_renewables(iso_code, simplify=False, save=False):
    """
    Fit wind and solar data for country, save parameters and copula for country.
    """
    # Read data
    wind_data = pd.read_csv("../data/WindNinja/renewables_ninja_europe_wind_output_1_current.csv", index_col=0)
    pv_data = pd.read_csv("../data/PVNinja/ninja_pv_europe_v1.1_merra2.csv", index_col=0)
    wind_data.index = pd.to_datetime(wind_data.index, format="%d/%m/%Y %H:%M")
    pv_data.index = pd.to_datetime(pv_data.index, format="%Y-%m-%d %H:%M:%S")

    # Fit wind data
    country = iso_to_country[iso_code]
    if country not in wind_data.columns:
        raise ValueError(f"{iso_code} is not a valid country code")
    wind_data_formatted = date_format(wind_data, country)
    wind_params, wind_covariance = fit_multivariate_weib(wind_data_formatted, simplify)

    # Fit solar data
    pv_data_formatted = date_format(pv_data, iso_code)
    pv_params, pv_covariance, night_hours = fit_multivariate_beta(pv_data_formatted, simplify=True)

    # Save results
    if save:
        wind_params.to_csv(f"parameters/marginals-wind-{country}")
        if simplify:
            wind_covariance = dia_matrix(wind_covariance)
        save_npz(f"parameters/copula-wind-{country}", wind_covariance)

        pv_params.to_csv(f"parameters/marginals-PV-{country}")
        pd.DataFrame(night_hours, columns=["day", "hour"]).to_csv(f"parameters/night_hours-PV-{country}")
        if simplify:
            pv_covariance = dia_matrix(pv_covariance)
        save_npz(f"parameters/copula-PV-{country}", pv_covariance)

    return wind_params, wind_covariance, pv_params, pv_covariance

#%%

all_params = fit_renewables("IT")
#%%
def fit_Ninja_all():
    """
    Fit wind and solar data for multiple countries, save parameters and copulas for each country.
    """
    #%%
    wind_pu = pd.read_csv("../WindNinja/renewables_ninja_europe_wind_output_1_current.csv", index_col = 0)
    PV_pu = pd.read_csv("../PVNinja/ninja_pv_europe_v1.1_merra2.csv", index_col = 0)
    wind_pu.index = pd.to_datetime(wind_pu.index, format="%d/%m/%Y %H:%M")
    PV_pu.index = pd.to_datetime(PV_pu.index, format = "%Y-%m-%d %H:%M:%S")
    path = "data.xlsx"
    EL=pd.read_excel(path,sheet_name='Electricity Load')
    GL=pd.read_excel(path,sheet_name='Gas Load')
    S=pd.read_excel(path,sheet_name='Solar')
    W=pd.read_excel(path,sheet_name='Wind')
    # %% fit wind for country
    countries = wind_pu.columns
    
    n_countries = len(countries)
    simplify = True
    
    
    
    
    PV_countries = list(PV_pu.columns) #extended name of countries
    
    countries_in_common = [iso_to_country[x] for x in PV_countries if iso_to_country[x] in countries]
    
    #%%
    
    c = 0
    for country in countries[10:]:
        print(f"Perc: {c/n_countries*100}%, fitting {country}")
        X = date_format(wind_pu, country)
        params, E_h = fit_multivariate_weib(X, simplify=True)
        params.to_csv(f"parameters2/marginals-wind-{country}")
        if simplify:
            E_h = dia_matrix(E_h)
            save_npz(f"parameters2/copula-wind-{country}",E_h)
        #pd.DataFrame(E_h).to_csv(f"parameters/copula-wind-{country}", float_format='%.3f')
        c += 1
        
    
    #params_dict = {} #for each country
    #%% fit solar
    c=0
    for pv_country in PV_countries:
        country = iso_to_country[pv_country]
        print(f"Perc: {c/n_countries*100}%, PV fitting {country}")
        X = date_format(PV_pu, pv_country)
        params, E_h, night_hours= fit_multivariate_beta(X, simplify=True)
        params.to_csv(f"parameters2/marginals-PV-{country}")
        pd.DataFrame(night_hours, columns = ("day","hour")).to_csv(f"parameters/night_hours-PV-{country}")
        if simplify:
            E_h = dia_matrix(E_h)
            save_npz(f"parameters2/copula-PV-{country}",E_h)
            
        c += 1  
        
    
    #%%
    E = E_h
    E_zoom = E[0:24*3,0:24*3]
    c= plt.imshow(E_zoom, interpolation = "nearest")
    plt.colorbar(c)
    plt.title("Gaussian copula covariance matrix")
    
    # %%
    X_PV = date_format(PV_pu,"AL")
    #%%
    
    parameters_df, E_h, night_hours = fit_multivariate_beta(X_PV)
    #%%
    n_scenarios = 10
    var_names = parameters_df.columns
    
    #%% load scenarios
    
    #demand = pd.read_csv("../DemandEnergy/time_series_60min_singleindex.csv")
    wind_scenarios = pd.read_csv("scenarios/wind_scenarios.csv", index_col=0)
    PV_scenarios = pd.read_csv("scenarios/PV_scenario100.csv", index_col=0)
    
    #%% format demand from df
    n_scenarios = 100
    n_hours = 24*365
    location = "1_i"
    season = "Q1"
    df = EL
    column = "Load"
    location_column = "Location_Electricity"
    m_to_s = [f"Q{month%12// 3 + 1}" for month in range(1, 13)] #month to season
    EL=pd.read_excel(path,sheet_name='Electricity Load')
    
    
    
    
    #%% # save electiricty: 
    locations = EL["Location_Electricity"].unique()
    for location in locations:
        quarters_df_to_year("electric-demand", location, EL, column, location_column, save = True, n_scenarios = 100, n_hours = 24*365)
    
    #%%
    quarters_df_to_year("hydrogen-demand", "g", GL, column, "Location_Gas", save = True, n_scenarios = 100, n_hours = 24*365)
    
    
    
#%%
    
if __name__ == "__main__":
    scenarios = {}
    scenarios["PV"]=pd.read_csv('scenarios/PV_scenario100.csv',index_col=0)
    scenarios["PV"].columns = pd.to_datetime(scenarios["PV"].columns)
    scenarios["wind"]=pd.read_csv('scenarios/wind_scenarios.csv',index_col=0)
    scenarios["wind"].columns = pd.to_datetime(scenarios["PV"].columns)
    params = read_parameters("Italy")
    #%%
    fig, ax = plt.subplots(figsize=(6, 6))
    c_pv = ax.imshow(params["PV"][1][0:24*3, 0:24*3], interpolation="nearest", aspect='auto')
    fig.colorbar(c_pv, ax=ax)
    ax.set_title("Covariance of PV values at different hours")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Hour")
    #plt.savefig('images/covariance_PV.png')
    #plt.close()
    #%%
    # Subplot for Covariance of Wind values at different hours
    fig, ax = plt.subplots(figsize=(6, 6))
    c_wind = ax.imshow(params["wind"][1][0:24*3, 0:24*3], interpolation="nearest", aspect='auto')
    fig.colorbar(c_wind, ax=ax)
    ax.set_title("Covariance of Wind values at different hours")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Hour")
    #plt.savefig('images/covariance_wind.png')
    #plt.close()

# %%

    params["wind"][1]
    #L = np.linalg.cholesky(params["wind"][1])
    #wind_scenarios = SG_weib(100, params["wind"], L, precomputed_decomposition = True)

    # %%
   
    #%%
    # Subplot for PV Power Output through the Year
    # wind_scenarios = scenarios["wind"]
    # PV_scenarios = scenarios["PV"]
    # n_scenarios = wind_scenarios.shape[0]
    
    # max_iter = 3  # plot at most 5 scenarios
    # wind_scenarios = pd.DataFrame(wind_scenarios.iloc[:max_iter, :])
    # PV_scenarios = pd.DataFrame(PV_scenarios.iloc[:max_iter, :])
    # colormap = cm.get_cmap('tab10', 2 * max_iter)
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for index, row in enumerate(wind_scenarios.iterrows()):
    #     ax.plot(row[1].index, row[1].values, color=colormap(index), alpha=0.7)
    # ax.set_title("Wind Power through the Year")
    # ax.set_xlabel("Datetime")
    # ax.set_ylabel("Power Output (capacity factor)")
    # ax.legend([f"Scenario {i+1}" for i in range(max_iter)], loc='upper right')
    # plt.savefig('images/wind_power_output_year.png')
    # plt.close()
    # #%%
    # # Subplot for PV Power Output through the Year
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for index, row in enumerate(PV_scenarios.iterrows()):
    #     ax.plot(row[1].index, row[1].values, color=colormap(max_iter + index), alpha=0.7)
    # ax.set_title("PV Power Output through the Year")
    # ax.set_xlabel("Datetime")
    # ax.set_ylabel("Power Output (capacity factor)")
    # ax.legend([f"PV Power {i+1}" for i in range(max_iter)], loc='upper right')
    # plt.savefig('images/PV_power_output_year.png')
    # plt.close()
    # #%%
    # # Subplot for 3 Days of Wind Power Output
    # day = 7*24
    # dday = 5
    # wind_scenario = wind_scenarios.iloc[0:max_iter, day*24:24*(day + dday)]
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for index, row in enumerate(wind_scenario.iterrows()):
    #     ax.plot(row[1].index, row[1].values, color=colormap(index), alpha=0.7)
    # ax.set_title("Wind Output for Three Days")
    # ax.set_xlabel("Datetime")
    # ax.set_ylabel("Power Output (capacity factor)")
    # ax.legend([f"Wind Power {i+1}" for i in range(max_iter)], loc='upper right')
    # #plt.savefig('images/wind_power_output_3days.png')
    # #plt.close()
    # #%%
    # solar_scenario = PV_scenarios.iloc[0:, day*24:day*24+24*dday]
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for index, row in enumerate(solar_scenario.iterrows()):
    #     ax.plot(row[1].index, row[1].values, color=colormap(max_iter + index), alpha=0.7)
    # ax.set_title("PV Output for Three Days")
    # ax.set_xlabel("Datetime")
    # ax.set_ylabel("Power Output (capacity factor)")
    # ax.legend([f"PV Power {i+1}" for i in range(max_iter)], loc='upper right')
    #plt.savefig('images/PV_power_output_3days.png')
    #plt.close()
