# Renewable Energy Scenario Generation and Optimization

This repository contains scripts for generating scenarios of wind and solar power output and optimizing an electrical grid supported by hydrogen storage. The scenario generation process considers the stochastic behavior of wind and solar power, and the optimization ensures the grid's reliability under various scenarios.

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/renewable-energy-scenarios.git
cd renewable-energy-scenarios
```
2. Install the required dependencies
```
pip install -r requirements.txt
```

### Dependencies

- gurobipy==11.0.1
- matplotlib==3.8.0
- numpy==1.26.4
- pandas==2.2.2
- pypsa==0.27.0
- PyQt6==6.7.0
- PyQt6_sip==13.6.0
- reliability==0.8.16
- scipy==1.13.0


## Scripts

### 1. `RenewableGridApp.py`

This script contains the implementation of a PyQt-based graphical user interface (GUI) for generating scenarios and optimizing the electrical grid.

#### Key Features:
- Provides a user-friendly interface for inputting parameters and generating scenarios.
- Displays the generated scenarios and optimization results graphically.
- Utilizes threading to ensure the GUI remains responsive during long computations.
- Cost and social acceptance parameters are easily costumizable

#### Usage:
Run the script using Python:
```bash
python test1gui.py
```

### 2. scenario_generation.py

This script focuses on the scenario generation process for wind and solar power output.
Key Features:

  - Models the power output of wind turbines and solar panels using Weibull and Beta distributions, respectively.
  - Utilizes Gaussian copulas to capture the dependencies between hourly power outputs.
  - Generates realistic scenarios based on historical data and fitted distributions.

Functions:

  - fit_weibull: Fits a Weibull distribution to wind power data.
  - fit_multivariate_weib: Fits a Couples multiple weibull distributions with Gaussian Copula. It take as input a pandas DataFrame where columns correspond to variables to be coupled and rows to samples. It return the parameters for the Weibull marginals and the empirical covariance matric of the Gaussian Copula.
  - SG_weib: Generates scenarios using the fitted Weibull parameters and copula.
  - fit_beta: Fits a Beta distribution to solar power data.
  - fit_multivariate_beta: Couples Beta distributions with a Gaussian copula.
  - SG_beta: Generates scenarios using the fitted Beta parameters and copula.


### 3. OPT.py

opt.py is a Python script designed to perform optimization of energy system components using the Gurobi optimization library. The script optimizes the configuration of energy sources and storage to minimize costs or maximize efficiency, considering various constraints and parameters.
Features

  - Load and preprocess energy system data from an Excel file.
  - Define and solve optimization problems using Gurobi.
  - Save and export optimization results to CSV files.
  - Plot results to visualize the optimization outcomes.

Functions:

  -load_data(filepath): Loads data from the specified Excel file.
  -OPT(ES, EW, EL, HL, cs, cw, mw, ch, chte, fhte, Mhte, ceth, feth, Meth): Defines and solves the optimization problem. ES, EW, EL and HL correspond to scenarios datasets respectively for Solar Power, Wind Power, Electric Load and Hydrogen Load. The columns correspond to timesteps and rows to scenarios. 
  -run_OPT(cs=4000, cw=3000000, mw=100, ch=10000, chte=0, fhte=0.75, Mhte=200000, ceth=0, feth=0.7, Meth=15000): Loads data and runs the optimization with default or specified parameters.
