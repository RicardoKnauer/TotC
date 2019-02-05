import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from model_TotC import *

problem = {
    'num_vars': 2,
    'names': ['initial_herdsmen', 'initial_sheep_per_herdsmen'],
    'bounds': [[1, 10], [0, 5]]
}

replicates = 10
max_steps = 100
distinct_samples = 20

model_reporters = {"Grass": lambda m: m.get_grass_count(),
                   "Sheep": lambda m: m.get_sheep_count()}

data = {}

for i, var in enumerate(problem['names']):
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)

    batch = BatchRunner(TotC, 
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var: samples},
                        model_reporters=model_reporters,
                        display_progress=True)    
    batch.run_all()
    data[var] = batch.get_model_vars_dataframe()

# Running  ^^^^^
# Plotting vvvvv

def plot_param_var_conf(ax, df, var, param, i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)

def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    f, axs = plt.subplots(3, figsize=(7, 10))
    
    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)

for param in ('Grass', 'Sheep'):
    plot_all_vars(data, param)
    plt.show()