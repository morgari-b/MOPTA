# Renewable Energy Scenario Generation and Optimization

This repository contains scripts for generating scenarios of wind and solar power output and optimizing an electrical grid supported by hydrogen storage. The scenario generation process considers the stochastic behavior of wind and solar power, and the optimization ensures the grid's reliability under various scenarios.

### Installation

    git clone https://github.com/yourusername/renewable-energy-scenarios.git2. prova_bianca.py









## Scripts

### 1. `test1gui.py`

This script contains the implementation of a PyQt-based graphical user interface (GUI) for generating scenarios and optimizing the electrical grid.

#### Key Features:
- Provides a user-friendly interface for inputting parameters and generating scenarios.
- Displays the generated scenarios and optimization results graphically.
- Utilizes threading to ensure the GUI remains responsive during long computations.

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
  - fit_multivariate_weib: Fits a Couples multiple weibull distributions with Gaussian COpula.
  - SG_weib: Generates scenarios using the fitted Weibull parameters and copula.
  - fit_beta: Fits a Beta distribution to solar power data.
  - fit_multivariate_beta: Couples Beta distributions with a Gaussian copula.
  - SG_beta: Generates scenarios using the fitted Beta parameters and copula.




Installation

    Clone the repository:
