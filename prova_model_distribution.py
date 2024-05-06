import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import erf, sqrt

def phi(x, mu, var):
    """
    Cumulative distribution function for normal distribution.
    x: real value
    mu: mean
    var: variance
    """
    return (1 + erf((x - mu) / sqrt(2 * var))) / 2.0

# Generate random dataset.
mean = np.linspace(30, 50, 24)
A = np.random.random((24, 24))
covar = np.dot(A, A.transpose())  # Ensure covariance matrix is symmetric and positive semi-definite

# Plot setup
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Subplots in a 2x3 grid
fig.suptitle("Copula multivariate estimation overview")


c = axes[0,0].imshow(covar, interpolation='nearest')
fig.colorbar(c, ax=axes[0,0])
axes[0,0].set_title("Multivariate Gaussian Covariance Matrix")


# Create random multivariate normal data
Y = np.random.multivariate_normal(mean, covar, 1000)
time_range = pd.date_range(start="20210101", end="20210102", periods=24)
Y = pd.DataFrame(Y, columns=time_range)


# Plot the first variable's histogram
axes[0, 1].hist(Y.iloc[:, 0], bins=20, color='skyblue', edgecolor='black')
axes[0, 1].set_title("Marginal distribution of var0")

# Parametric estimation of marginal distributions
mean_est = Y.mean(axis=0)
scale_est = np.sqrt(Y.var(axis=0, ddof=1))


#uniform variables estimation
X_h = pd.DataFrame(index=Y.index, columns=Y.columns)
O_h = pd.DataFrame(index=Y.index, columns=Y.columns)
for var in Y.columns:
    X_h[var] = norm.cdf(Y[var], loc=mean_est[var], scale=scale_est[var])
    O_h[var] = norm.ppf(X_h[var])
# Histogram of transformed data to uniform distribution using estimated CDF
axes[0, 2].hist(X_h.iloc[:, 0], bins=20, color='lightgreen', edgecolor='black')
axes[0, 2].set_title("Uniform distrib by composing data via estimated CDF") #se non sembra uniforma not a good estimation

axes[1, 0].hist(O_h.iloc[:, 0], bins=20, color='lightgreen', edgecolor='black')
axes[1, 0].set_title("this should be gaussian") #se non sembra uniforma not a good estimation

#copula covariance estimation:
E_h = np.cov(O_h.T, ddof = 1)

#plot
#c = axes[1,1].imshow(E_h)
#fig.colorbar(c, ax = axes[1,1])
#axes[1,1].set_title("copula Covariance Matrix")
   

### GENERATE SCENARIOS
#copula_scenarios
n_scenarios = 1000
n_vars = len(Y.columns)
X_scen = pd.DataFrame(np.random.multivariate_normal(np.array([0]*n_vars), E_h, n_scenarios), columns = Y.columns)
#convert to uniform domain U and the to inidital domain Y
U_scen = pd.DataFrame( columns=Y.columns)
scenarios = pd.DataFrame( columns=Y.columns) #scenario dataframe

for var in Y.columns:
    U_scen[var] = norm.cdf(X_scen[var])
    scenarios[var] = norm.ppf(U_scen[var], loc = mean_est[var], scale = scale_est[var])


# Plot the first variable's histogram
axes[1, 1].hist(scenarios.iloc[:, 0], bins=20, color='skyblue', edgecolor='black')
axes[1, 1].set_title("approx Marginal distribution of var0")

scenarios_covariance = np.cov(scenarios.T, ddof = 1)
c = axes[1,2].imshow(scenarios_covariance)
fig.colorbar(c, ax = axes[1,2])
axes[1,2].set_title("scenarios_covariance matrix")
plt.tight_layout()
plt.show()
