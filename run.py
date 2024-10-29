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
parser.add_argument("--method")
parser.add_argument("--seed")
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
n_init = 3 if args.problem == "berkenkamp" else 10  # number of initial random samples
n_iterations = 250                                   # numer of BO iterations
regret_type = "lengthscale"                         # regret type, ['lengthscale', ]

# method parameters
method = args.method  # method ["LB-GP-UCB", "random", "MLE", "berkenkamp"]
tol=1e-2              # tolerance parameter
gamma=20              # gamma parameter
R=1e-4                # noise variance (likelihood)
n_hypers_start=5      # number of hyperparameters at the beginning
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

if args.gt == "sqrtinit":
    init = math.exp(n_hypers_start/(train_X.shape[1] + 2*2.5))
    g = lambda t: max(init, t**0.5)
elif args.gt == "25init":
    init = math.exp(n_hypers_start/(train_X.shape[1] + 2*2.5))
    g = lambda t: max(init, t**0.25)
elif args.gt == "75init":
    init = math.exp(n_hypers_start/(train_X.shape[1] + 2*2.5))
    g = lambda t: max(init, t**0.75)


hm = HyperparameterManager(
    model,
    g,
    regret_type=regret_type,
    delta=delta,
    C=C,
    tol=tol,
    R=R,
    kernel=args.kernel
)
hypersamples = hm.initialise_setup(model)
# upper limit of lengthscale
theta_0 = hm.theta_H

if args.method == "LB-GP-UCB" or args.method == "cBerkenkamp":
    args.method += f"C={C}_n_hypers_start={n_hypers_start}_gt={args.gt}"

args.method += f"_{args.kernel}"
min_ls = float("inf")

if method == "oracle":
    oracle_dict = {
        "michalewicz5":0.035411037051703544,
        "holder": 0.05077924561135193,
        "crossedbarrel": 0.13089737431009674,
        "berkenkamp": 0.004377865013756788,
        "agnp": 0.10563971624659162
    }
    oracle_theta = oracle_dict[args.problem]

save_path = os.path.join("runs", args.problem, args.method, args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

log = pd.DataFrame()

# running BO
results = []
cumulative_y = 0.0
for t in range(1, n_iterations+1):
    tik = time.monotonic()
    if method == "random":
        X_next = domain.sample(1)
    else:
        model.conditioning(train_X, train_Y)

        # Training: select benchmark method
        if method == "LB-GP-UCB": 
            # Training: select hyperparameter that has the minimum regret
            idx_umin_reduced = hm.argmin_regret(hypersamples)
            hyper = hypersamples[idx_umin_reduced]
            model.set_hyper(hyper, hypersamples)
        elif method in ["cBerkenkamp", "cMLE"]:
            model.train(hypersamples, t=t, g=g, theta_0=theta_0)
        elif method == "oracle":
            model.train(hypersamples, oracle_theta=oracle_theta)
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

    min_ls = min(min_ls, min(model.gp.covar_module.base_kernel.raw_lengthscale_constraint.transform(hypersamples[:,1])).item())

    log = pd.concat(
        [log,
         pd.DataFrame(
             {
                 'y': [Y_next.item()],
                 'cumulative_y': [cumulative_y],
                 'best_y': [best_obs.item()],
                 't': [t],
                 'overhead': [overhead],
                 'lengthscale': [model.gp.covar_module.base_kernel.lengthscale.detach().item()],
                 'seed': [int(args.seed)],
                 'min_ls': [min_ls],
                 'no_hypers': [len(hm.S)]
             }
         )]
    )

    log.to_csv(os.path.join(save_path, "log.csv"))

    if method == "LB-GP-UCB":
        beta = acqf.beta
        _, sigma = model.predictive_mean_and_stddev(X_next)
        hypersamples = hm.update_hyperparameters(model, hypersamples, t, idx_umin_reduced, Y_next, beta, sigma)
        hypersamples = hm.add_new_hyperparameters(hypersamples, model, t+1)