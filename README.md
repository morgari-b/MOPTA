# Renewable Energy Scenario Generation and Optimization

Team name: Clowder
This repository contains scripts for generating scenarios of wind and solar power output and optimizing an electrical grid supported by hydrogen storage. The scenario generation process considers the stochastic behavior of wind and solar power, and the optimization ensures the grid's reliability under various scenarios. To tackle computational complexity of the problem, a time aggregation method is introduced.

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/MOPTA
cd MOPTA
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

### 1. `App.py`

This script contains the implementation of a Plotly-based graphical user interface (GUI) for generating scenarios and optimizing the electrical grid.

#### Key Features:
- Provides a user-friendly interface for creating and Energy Grid and generating scenarios.
- Displays the generated scenarios and optimization results graphically.
- Utilizes threading to ensure the GUI remains responsive during long computations.
- Cost and social acceptance parameters are easily costumizable

#### Usage:
Run the script using Python:
```bash
python App.py
```

### 2. model/scenario_generation

This folder constains all the scripts necessary for the fitting of stochastic random variables related to the grind and for generating scenarios.

#### 2.1 scenario_generation.py

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

#### 2.2 parameters_countries

Folder containing the fitted parameters for wind and PV distributions in european countries. the files starting with "copula" contain the covariance matrix of the Guassian copula of the corresponding multidimensional variable. Files starting with "marginals" containg the parameters describing the Weibull and Beta distributions describing Wind an PV variables respectively. "night_hours" files record hours during the year in which there is no PV production.
### 3. model

the "model" folder contains the scripts to model the network and time partitions and the various optimization methods employed in the app.


#### 3.3 YUPPY.py
YUPPY.py defines the class Network to model the energy grid and the class time_partition to keep track of the time partitions iterations.

#### 3.2 OPT_methods,py

OPT_methods.py is a Python script designed to perform optimization of energy system components using the Gurobi optimization library. The script optimizes the configuration of energy sources and storage to minimize costs or maximize efficiency, considering various constraints and parameters.
Features

  - Load network object as de
  - Define and solve optimization problems using Gurobi.
  - Save and export optimization results to CSV files.
  - Plot results to visualize the optimization outcomes.

Optimization methods:

  -OPT3: takes as input a network and optimizes it using without any time aggragration. This is equivalent to solving with the finest time partition. This is the most accurate but also slowest model.
  -OPT_agg: takes as input a network with a define time_partition and solve the corresponding relaxed method.
  -OPT_time_partition: iteratevely optimizes over finer time partitions leveraging the warm start method: corresponds to a constraints and column aggregation with same weight.
  -OPT_agg2: similar to OPT_time_partition but we add all the variables in the beginning, and then add aggregated time constraints. At each iteration we add the time constraints corresponding to the refined intervals. This is a constraint aggregation method.
    Note: OPT_agg2 only works when disaggregating intervals into singletons, it doesn't work when splitting intervals into smaller intervals of lenght greater than one.

  #### 3.3 OPloTs.py

  Contains the functions for plotting the results of the optimizaiton methods in OPT_methods.py


  #### 3.4 EU_net.py

  Contains "EU", a small example network of five nodes in the European Union.
