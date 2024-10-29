import argparse
import os
import torch
import copy
import time
import pyro
import pandas as pd
from utils._main import initial_conditions
from utils._hb_ucb import HyperparameterManager
from utils._experiments import select_experiment
from utils._acquisition_function import AcquisitionFunction
from utils._benchmarks import continuous_MLE, continuos_berkenkamp

parser = argparse.ArgumentParser()
parser.add_argument("--problem")
parser.add_argument("--seed")
parser.add_argument("--kernel")
args = parser.parse_args()

torch.manual_seed(args.seed)
pyro.set_rng_seed(args.seed)

test_function, domain = select_experiment(
    args.problem,               # function to maximize
    noise_std=None,             # noiseless feedback
    dtype=torch.float64,        # double or float precision
)

# experimental parameters
seed = args.seed                                    # random seed
n_init = 3 if args.problem == "berkenkamp" else 10  # number of initial random samples
n_iterations = 50                                   # numer of BO iterations
regret_type = "lengthscale"                         # regret type, ['lengthscale', ]

# method parameters
method = "FullyBayesianGP"
tol=1e-2              # tolerance parameter
gamma=20              # gamma parameter
R=1e-4                # noise variance (likelihood)
n_hypers=5            # number of hyperparameters, |U|
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
    kernel_type=args.kernel
)

save_path = os.path.join("runs", args.problem, args.method, args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

log = pd.DataFrame()

# running BO
hypersamples = []
results = []
cumulative_y = 0.0
for t in range(1, n_iterations+1):
    tik = time.monotonic()
    model.conditioning(train_X, train_Y)

     # Training: select benchmark method
    if method == "FullyBayesianGP":
        model.train()
    else:
        raise ValueError(f"No such method as {method}.")
    
    acqf = AcquisitionFunction(model, n_iterations, tol=tol, gamma=gamma, R=R, method=information_gain)
    X_next = acqf.optimize(t)
    tok = time.monotonic()
    overhead = tok - tik

    Y_next = test_function(X_next)

    train_X = torch.vstack([train_X, X_next])
    train_Y = torch.cat([train_Y.squeeze(), Y_next])
    if hasattr(test_function, "remainig_index"):
        train_Y = train_Y.unsqueeze(-1)
    best_obs = train_Y.max()
    print('Iter %d - overhead: %.3f   best found: %.3f' % (t, overhead, best_obs))
    results.append([overhead, best_obs])
    cumulative_y += Y_next.item()

    log = pd.concat(
        [log,
         pd.DataFrame(
             {
                 'y': [Y_next.item()],
                 'cumulative_y': [cumulative_y],
                 'best_y': [best_obs.item()],
                 't': [t],
                 'overhead': [overhead],
                 'lengthscale': [model.gp.covar_module.base_kernel.lengthscale.mean().detach().item()],
                 'seed': [int(args.seed)]
             }
         )]
    )

    log.to_csv(os.path.join(save_path, "log.csv"))