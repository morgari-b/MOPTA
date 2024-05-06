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

def fit_weibull(Y):
    """
    Y : pandas dataframe having as columns variables and as rows scenarios
    returns dataframe having estimated with MLE weibull parameters.
    """
    parameters_df = pd.DataFrame(columns = Y.columns)
    for var in Y.columns:
        params = stats.exponweib.fit(Y[var], method = "MLE")
        parameters_df.loc["scale", var],parameters_df.loc["shape", var] = params[3], params[1]
        
    return parameters_df

def fit_multivariate_weib(Y):
    """
    Fit data with multivariate distribution having Weibull marginals coupled with Gaussian copula.
    input
        Y: Pandas dataframe taking having as columns variables and rows as samples
    output
        parameters_df: df having for each variable the estimated parameters of the Weibull distribution
        E_h: the estimated covariance matrix of the Gaussian Copula
        
    """
    #weibul distribution estimation
    parameters_df = fit_weibull(Y)

    #copula estimation
    
    X_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    O_h = pd.DataFrame(index=Y.index, columns=Y.columns)
    for var in Y.columns:
        #bring data to uniform domain
        X_h[var] = stats.exponweib.cdf(Y[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        O_h[var] = stats.exponweib.ppf(X_h[var], scale = parameters_df.loc["scale",var],a = 1, c = parameters_df.loc["shape",var])
        
    #copula covariance estimation:
    E_h = np.cov(O_h.T, ddof = 1)
    
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

def SG_weib_example(n_samples = 100, n_scenarios = 100, n_vars = 24, shape = 2, scale = 3 ):
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

    axes[0,0].hist(Y.iloc[:,0], bins = 20)
    axes[0,0].set_title("Data Histogram")

    parameters_df, E_h = fit_multivariate_weib(Y)
    
    ax = axes[0,1]
    c = ax.imshow(E_h, interpolation = "nearest")
    fig.colorbar(c, ax=ax)
    ax.set_title("Gaussian copula covariance matrix")
    
    ax = axes[0,2]
    ax.plot(parameters_df.loc["scale"])
    scenarios = SG_weib(100,Y.columns, parameters_df, E_h)



    axes[1,0].hist(scenarios.iloc[:,0], bins = 20)
    axes[1,0].set_title("Data Scenarios")

    plt.show()
