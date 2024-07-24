#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:06:23 2024

@author: frulcino
"""

import pandas as pd
import matplotlib.pyplot as plt

columns = pd.date_range("01/01/2023", periods = 24*365, freq = "h")
eth = pd.read_csv("Outputs/eth.csv", names = columns, index_col = False)
hte = pd.read_csv("Outputs/hte.csv", names = columns, index_col = False)
hh = pd.read_csv("Outputs/hh.csv", names = columns, index_col = False)
outputs = pd.read_csv("Outputs/outputs.csv", names = ["ns","nw","nh","mhte","meth"], index_col = False)

