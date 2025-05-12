import pypsa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandapower.networks as pn
#import math
from gurobipy import Model, GRB, Env, quicksum
import time
from itertools import product
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, AutoDateLocator, ConciseDateFormatter #mdates
import os
#%% os.chdir("C:/Users/ghjub/codes/MOPTA/02_model")
import xarray as xr
import folium
from model.scenario_generation.scenario_generation import import_generated_scenario, import_scenario, scenario_to_array
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class time_partition:

    def aggregate(self,l, i0,i1):
        """
        Aggregates a list into sublists based on the given indices.

        Args:
            l (list): The list to be aggregated.
            i0 (int): The starting index of the sublist.
            i1 (int): The ending index of the sublist.

        Returns:
            list: The aggregated list with sublists.
        """
        if i0 == 0:
            return [l[i0:i1]]+ l[i1:]
        elif i1 == len(l):
            return l[0:i0]+[l[i0:i1]]
        elif i0 == i1:
            return l
        else:
            return l[0:i0]+[l[i0:i1]]+ l[i1:]
    def disaggregate(self,l,i):
        if type(l[i]) is list:
            return l[0:i] + l[i] + l[i+1:]
        else:
            raise ValueError(f"{l[i]} is not a list and cannot be disaggregated")

    def day_aggregation(self):
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        for day in range(n_days): #ragruppo i giorni
            l = self.aggregate(l, day, day + 24)

        n_seasons = 16
        season = int(np.floor(n_days / n_seasons))  #mettiamo 10 giorni interi per intero
        for i in range(n_seasons):
            l = self.disaggregate(l, i * season + 23*i)
        return l

    def day_night_aggregation(self):
        """
        Aggregates the time steps into sublists based on day and night periods.

        Returns:
            list: The aggregated list with sublists representing day and night periods.
        """
        l = self.time_steps
        n_days = int(np.floor(self.T / 24))
        l = self.aggregate(l, 0, 6) #initial night
        for day in range(n_days-1): #ragruppo i giorni
            l = self.aggregate(l, 2*day + 1, 2*day + 1 + 14)
            l = self.aggregate(l, 2*day + 2, 2*day + 2 + 10) 
        day = n_days-1
        l = self.aggregate(l, 2*day + 1, 3*day + 1 + 14)
        l = self.aggregate(l, 2*day + 2, 3*day + 2 + 4)
        l1 = []
        for i in l:
            if type(i) is int: #ho modificato qui protrebbe creare problemi, stavo cercando di togliere liste vuote.
                l1 += [i]
            elif len(i) > 1:
                l1.append(i)
        l = l1
        #season = int(np.floor(n_days /10))  #mettiamo 10 giorni interi per intero
        #for i in range(10):
        #    l = self.disaggregate(l, i * season + 23*i)
        return l

    def __init__(self,T, init_method='day_aggregation'):
        self.T=T
        self.time_steps = list(range(T))
        #print(init_method)
        #define initial_aggregation with the chosen method.
        method = getattr(self, init_method)
        if method is not None and callable(method):
            self.agg = method()
        else:
            print(f"Method {init_method} not found or not callable")
        
        self.old_agg = []
        self.family_tree = [] #list of lists of intervals splitted in some way (todo? become a dictionary mapping which interval in splitted into which intervals)

    def len(self,i):
        if type(self.agg[i]) is list:
            return len(self.agg[i])
        else:
            return 1

    def random_iter_partition(self,k=1):
        self.old_agg += [self.agg.copy()]
        family_list = []
        for _ in range(k):
            tp = self.agg
            agg_indices = [i for i in range(len(tp)) if type(tp[i]) is list] #get index of aggregate intervals
            rand_ind = agg_indices[np.random.randint(len(agg_indices))] #get random index
            new_int =self.agg[rand_ind]
            self.agg = self.disaggregate(self.agg,rand_ind)
            family_list += [new_int]
        self.family_tree += [time_partition.order_intervals(family_list)]

    def iter_partition(self, t):
        self.old_agg += [self.agg.copy()]
        family_list = []
        tp = self.agg
        new_int =self.agg[t]
        if type(new_int) is list:
            self.agg = self.disaggregate(self.agg,t)
            family_list += [new_int]
            self.family_tree += [time_partition.order_intervals(family_list)]
        else:
            print(f"{new_int} is a singleton")
    
    def iter_partition_intervals(tp_obj, intervals):
        "completly disaggregates intervals in intervals"
        tp_obj.old_agg += [tp_obj.agg.copy()]
        family_list = []
        intervals = sorted(intervals)[::-1] #reverse intervals order so that disaggregating does not change the indexes.
        for t in intervals:
            new_int =tp_obj.agg[t]
            if type(new_int) is list:
                tp_obj.agg = tp_obj.disaggregate(tp_obj.agg,t)
                family_list += [new_int]
            else:
                print(f"{new_int} is a singleton")
        if len(family_list) > 0:
            #print("appending new intervals to family tree")
            tp_obj.family_tree += [tp_obj.order_intervals(family_list)]
        else:
            print("No intervals where splitted, iteration left partition identical")
    def total_disaggregate(self):
        """
        Disaggregates all intervals in the time partition.

        Returns:
            list: The disaggregated list of intervals.
        """
        self.old_agg += [self.agg.copy()]
        to_disaggregate = []
        for i in range(len(self.agg)):
            new_int =self.agg[i]
           
            if type(new_int) is list:
                to_disaggregate += [i]
            else:
                #print(f"{new_int} is a singleton")
                continue
        if len(to_disaggregate) > 0:
            #print("appending new intervals to family tree")
            self.iter_partition_intervals(to_disaggregate)
        else:
            print("No intervals where splitted, iteration left partition identical")
    def to_dict(self):
        """
        Save the Network object as a dictionary of dictionaries, where each attribute is stored as a dictionary.

        Returns:
            dict: A dictionary containing all the attributes of the Network object.
        """
        output = {}
        for attr in self.__dict__:
            attribute = getattr(self, attr)
            if type(attribute) is int:
                output[attr] = attribute
            elif type(attribute) is float:
                output[attr] = attribute
            elif type(attribute) is str:
                output[attr] = attribute
            elif type(attribute) is list:
                output[attr] = attribute
            else:
                output[attr] = getattr(self, attr).to_dict()
        return output

    @staticmethod
    def from_dict(d):
        tp = time_partition(d['T'])
        for attr, value in d.items():
            setattr(tp, attr, value)
        return tp

    @staticmethod
    def tuplize(tp):
        l = []
        for i in tp:
            if type(i) is list:
                l.append(tuple(i))
            else:
                l.append(i)
        return l

    @staticmethod
    def partitionize(tp):
        l = []
        for i in tp:
            if type(i) is int:
                l.append([i])
            else:
                l.append(i)
        return l

    @staticmethod
    def order_intervals(L):
        return sorted(L, key=lambda l: l[0])

    @staticmethod
    def interval_subsets(interval,tp):
        """
        Get the indexes of elements in the given time partition `tp` that are subsets of the given `interval`.

        Parameters:
            interval (set): The set of elements to check for subset membership.
            tp (list): The list of elements to search for subsets of `interval`.

        Returns:
            list: The indexes of elements in `tp` that are subsets of `interval`.
        """
        indexes = []
        for i in range(len(tp)):
            l = tp[i]
            if type(l) is list:
                if set(l).issubset(interval):
                    indexes.append(i)
            else:
                if l in interval:
                    indexes.append(i)
        return indexes
        
# %% df_aggregator
