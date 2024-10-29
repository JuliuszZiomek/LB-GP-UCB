import copy
import json
import os
import random
import re
import subprocess
from typing import Any
import torch
import botorch
import pandas as pd
from ._domain import UniformDomain, FiniteDomain
import gpytorch
import math
from gpytorch.kernels import ScaleKernel, PeriodicKernel, MaternKernel
from ._domain import UniformDomain
#from ._synthetic_gp import SGP
#from ._kernels import OneComponentKernel
from gpytorch.constraints import Positive


def select_experiment(function, noise_std=None, dtype=torch.float64, seed=0):
    if function == "michalewicz5":
        return setup_michalewicz(n_dims=5)
    elif function == "berkenkamp":
        return setup_berkenkamp(noise_std=noise_std, dtype=dtype)
    elif function == "agnp":
        return setup_materials(dataset="AgNP")
    elif function == "crossedbarrel":
        return setup_materials(dataset="Crossed barrel")



def setup_michalewicz(n_dims=4, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([torch.zeros(n_dims), 3.14 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    test_function = botorch.test_functions.Michalewicz(
        dim=n_dims,
        negate=True,         # maximisation problem
    )
    return test_function, domain


def setup_berkenkamp(noise_std=None, dtype=torch.float64):
    class Berkenkamp:
        def __init__(self, noise_std=None):
            self.noise_std = noise_std
            self.hypersamples =  torch.tensor([
                [ 1.1492, -0.9342],
                [ 1.1492, -0.7342],
                [ 1.1492, -0.3342],
                [ 1.1492, 0.1342],
                [ 1.1492, 0.5342],
            ], dtype=dtype)
            
        def __call__(self, x):
            x = x.squeeze()
            y1 = 0.6 * x
            y2 = torch.distributions.Normal(0.2, 0.08).log_prob(x).exp() / 8
            y = torch.atleast_1d(y1 + y2)
            if not self.noise_std is None:
                y += torch.distributions.Normal(0, self.noise_std).sample((len(y),))
            return y
    
    bounds = torch.vstack([torch.zeros(1), torch.ones(1)]).to(dtype)
    domain = UniformDomain(bounds)
    test_function = Berkenkamp(noise_std)
    return test_function, domain

def setup_materials(dataset):
    try:
        raw_dataset = pd.read_csv("datasets/" + dataset + '_dataset.csv')
    except:
        raise FileNotFoundError(" Please download material datasets from https://github.com/PV-Lab/Benchmarking and put them to datasets/ directory.")
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]
    ds = copy.deepcopy(raw_dataset) 
    if dataset not in ['Crossed barrel']:
        ds[objective_name] = -raw_dataset[objective_name].values

    class MaterialsProblem():
        def __init__(self, ds):
            ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
            ds_grouped = (ds_grouped.to_frame()).reset_index()
            X_feature = ds_grouped[feature_name].values
            self.y = torch.tensor(ds_grouped[objective_name].values)

            self.y_max = torch.max(self.y)

            self.xdata = torch.tensor(X_feature)
            self.domain = FiniteDomain(self.xdata)
        
        def __call__(self, x) -> Any:
            out = []
            for row_x in x:
                out.append(torch.dot((row_x == self.xdata).all(-1).double(), self.y) - self.y_max)
            return torch.tensor(out).reshape(-1)
        
    f = MaterialsProblem(ds)
    return f, f.domain


