import copy
import os
import time
import torch
import pandas as pd
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from ._gaussian_process import GPEnsemble, SingleGP, FullyBayesianGP
from ._acquisition_function import AcquisitionFunction

def initial_conditions(
    seed,
    domain,
    test_function,
    method="MLE",
    kernel_type="RBF",
    generator="random",
    n_init=10,
    n_hypersamples=5,
    include_mle=False,
    lb=0.01,
    ub=0.5
):
    torch.manual_seed(seed)
    if hasattr(test_function, "remainig_index"):
        train_X, train_Y = domain.sample(n_init)
    else:
        train_X = domain.sample(n_init)
        train_Y = test_function(train_X)

    if hasattr(test_function, "mean_module"):
        mean_module = test_function.mean_module
    else:
        mean_module = None
    if hasattr(test_function, "covar_module"):
        covar_module = test_function.covar_module
    else:
        covar_module = None
    
    if kernel_type == "RBF":
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=None))
    elif kernel_type == "Matern":
        covar_module = ScaleKernel(MaternKernel(ard_num_dims=None))
    else:
        raise ValueError()
    
    if method in ["MLE", "random", "berkenkamp", "HB-GP-UCB", "cMLE", "cBerkenkamp", "cBerkenkamp_LB", "oracle"]:
        model = SingleGP(train_X, train_Y, domain, method=method, mean_module=mean_module, covar_module=covar_module)
    elif method in ["FullyBayesianGP"]:
        model = FullyBayesianGP(train_X, train_Y, domain, method=method, mean_module=mean_module, covar_module=covar_module)
    
    if hasattr(test_function, "hypersamples"):
        hypersamples = test_function.hypersamples
    else:
        hypersamples = None
    return train_X, train_Y, model, hypersamples
