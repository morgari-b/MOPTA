#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:09:24 2024

@author: frulcino

This scripts contains functions to model multivariate variables by:
    1) Fitting indipendently their marginal distributions
    2) Coupling the variables by fitting with a Gaussian Copula
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from reliability.Fitters import Fit_Weibull_2P
import os
os.chdir('/home/frulcino/codes/MOPTA/')


#TODOS
# - Can I use Y instead of O_h directly?
# %% helper functions

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


    
def fit_multivariate_weib(Y, method = "LS"):
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
        O_h[var] = stats.exponweib.ppf(X_h[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        
    #copula covariance estimation:
    E_h = np.cov(O_h.T, ddof = 1)
    
    
    c= plt.imshow(E_h, interpolation = "nearest")
    plt.colorbar(c)
    plt.title("Gaussian copula covariance matrix")
    
    return parameters_df, E_h

def SG_weib(n_scenarios, var_names, parameters_df, E_h):
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
    n_vars = len(var_names)
    X_scen = pd.DataFrame(np.random.multivariate_normal(np.array([0]*n_vars), E_h, n_scenarios), columns = var_names)
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

def fit_multivariate_beta(Y):
    """
    Fit data with multivariate distribution having Weibull marginals coupled with Gaussian copula.
    input
        Y: Pandas dataframe taking having as columns variables and rows as samples
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
        O_h[var] = stats.beta.ppf(X_h[var], alpha, beta)
        
    #copula covariance estimation:
    E_h = np.cov(O_h.T, ddof = 1) #non posso usare Y_h direttamente?
    

    c= plt.imshow(E_h, interpolation = "nearest")
    plt.colorbar(c)
    plt.title("Gaussian copula covariance matrix")
  
    return parameters_df, E_h, night_hours
#%%
def SG_beta(n_scenarios, var_names, parameters_df, E_h, night_hours, save = True, filename = "PV_scenario"):
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
#%% fetch data


wind_pu = pd.read_csv("../WindNinja/renewables_ninja_europe_wind_output_1_current.csv", index_col = 0)
PV_pu = pd.read_csv("../PVNinja/ninja_pv_europe_v1.1_merra2.csv", index_col = 0)
wind_pu.index = pd.to_datetime(wind_pu.index, format="%d/%m/%Y %H:%M")
PV_pu.index = pd.to_datetime(PV_pu.index, format = "%Y-%m-%d %H:%M:%S")

# %% format data for country
#countries = wind_pu.columns
#country = countries[0]

#X = date_format(wind_pu, country)


#params_dict = {} #for each country
#%% fit
#params = fit_multivariate_weib(X)

#%%
#E = params[1]
#E_zoom = E[0:24*7,0:24*7]
#c= plt.imshow(E_zoom, interpolation = "nearest")
#plt.colorbar(c)
#plt.title("Gaussian copula covariance matrix")

# %%
X_PV = date_format(PV_pu,"AL")
#%%

parameters_df, E_h, night_hours = fit_multivariate_beta(X_PV)
#%%
n_scenarios = 10
var_names = parameters_df.columns




