import argparse
import math
import os
import torch
import copy
import time
import pandas as pd
from utils._main import initial_conditions
from utils._hb_ucb import HyperparameterManager
from utils._experiments import select_experiment
from utils._acquisition_function import AcquisitionFunction
from utils._benchmarks import continuous_MLE, continuos_berkenkamp

parser = argparse.ArgumentParser()
parser.add_argument("--problem")
parser.add_argument("--seed", default=0)
parser.add_argument("--kernel", default="Matern")
parser.add_argument("--gt", default="exp")
args = parser.parse_args()

torch.manual_seed(args.seed)

test_function, domain = select_experiment(
    args.problem,               # function to maximize
    noise_std=None,             # noiseless feedback
    dtype=torch.float64,        # double or float precision
)

# experimental parameters
seed = args.seed                                    # random seed
n_init = 1000  # number of initial random samples
n_iterations = 100                                   # numer of BO iterations
regret_type = "lengthscale"                         # regret type, ['lengthscale', ]

# method parameters
method = "cMLE"
tol=1e-2              # tolerance parameter
gamma=20              # gamma parameter
R=1e-4                # noise variance (likelihood)
n_hypers_start=5      # number of hyperparameters at the beginning
n_hypers_end=5       # number of hyperparameters at the end
delta=0.01            # 0 < delta < 1. smaller delta leads to larger lower bound of lengthscale theta_L.
C=0.01                # constant in lengthscale regret
B=1                   # bound in regret
information_gain="exact"

# initial conditions
train_X, train_Y, model, hyperparameters = initial_conditions(
    seed,
    domain,
    test_function,
    method=method,
    n_init=n_init,
    kernel_type=args.kernel,
)
model.conditioning(train_X, train_Y)
model.train(hypersamples)
print(model.gp.covar_module.base_kernel.lengthscale.detach().item())
