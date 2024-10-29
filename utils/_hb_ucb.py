import copy
import math
import torch
import gpytorch
import botorch


class HypersampleGenerator:
    def maximum_likelihood_estimation(self, gp):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        botorch.fit.fit_gpytorch_model(mll)
        gp.eval()
        return gp
    
    def lengthscale_mle(self):
        self.gp = self.maximum_likelihood_estimation(self.gp)
        params = [param.item() for param_name, param in self.gp.named_parameters() if not "likelihood" in param_name]
        hyper_mle = torch.tensor(params, dtype=self.dtype)
        return hyper_mle
    
    def lengthscale_hypers(self):
        hyper_mle = self.lengthscale_mle()
        theta_H = self.constraint.transform(hyper_mle[self.idx_lengthscale])
        self.theta_H = theta_H
        new_theta = theta_H
        lengthscales = []
        while new_theta > self.theta_H / self.g(1):
            lengthscales.append(new_theta)
            self.theta_L = new_theta
            new_theta = new_theta /  self.ls_div
    
        lengthscales = torch.stack(lengthscales, dim=0)
        raw_lengthscales = self.constraint.inverse_transform(lengthscales)
        self.n_hypers = lengthscales.shape[0]
        
        hypersamples = hyper_mle.repeat(self.n_hypers,1)
        hypersamples[:, self.idx_lengthscale] = raw_lengthscales
        hypersamples[:,0] = self.gp.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(1)).item()
        return hypersamples
    
    def generate_hypers(self):
        if self.regret_type =="lengthscale":
            return self.lengthscale_hypers()


class RegretCalculator:
    def convert_lengthscale(self, hypersamples):
        lengthscale_raw = hypersamples[:, self.idx_lengthscale]
        lengthscales = self.constraint.transform(lengthscale_raw)
        return lengthscales
    
    def compute_xi(self, t):
        xi = 2 * self.R.pow(2) * (self.n_U * (torch.pi**2) * (t**2) / (6*self.tol)).log()
        return xi
    
    def regret_lengthscale(self, T, theta):
        theta_term = (1/theta)**self.n_dims
        logT_term = torch.tensor(T).to(self.dtype).log()**(self.n_dims + 1)
        first = self.C * (T * theta_term * logT_term).sqrt()
        second = (theta_term * logT_term).sqrt() +self.B
        return first * second
    
    def index_regret_lengthscale(self, hypersamples, idx, shift=0):
        lengthscales = self.convert_lengthscale(hypersamples)
        theta = lengthscales[idx]
        idx_original = self.remaining_indices[idx]
        T = self.n_S[idx_original] + shift
        regret = self.regret_lengthscale(T, theta)
        return regret
    
    def regret(self, hypersamples, index, shift=0):
        if self.regret_type =="lengthscale":
            regret = self.index_regret_lengthscale(hypersamples, index, shift=shift)
        return regret


class MemoryManager:
    def create_initial_memory(self):
        return [[] for i in range(self.n_U)]
        
    def initial_memory(self):
        self.S = self.create_initial_memory()
        self.n_S = [len(l) for l in self.S]
        self.Y = self.create_initial_memory()
        self.beta = self.create_initial_memory()
        self.sigma = self.create_initial_memory()

    def expand_memory(self):
        self.S.append([])
        self.n_S.append(len(self.S[-1]))
        self.Y.append([])
        self.beta.append([])
        self.sigma.append([])
    
    def update_memory(self, t, idx_umin_reduced, Y_next, beta, sigma):
        idx_umin = self.remaining_indices[idx_umin_reduced]
        self.S[idx_umin].append(t)
        self.n_S[idx_umin] += 1
        
        self.Y[idx_umin].append(Y_next)
        self.beta[idx_umin].append(beta.item())
        self.sigma[idx_umin].append(sigma.item())


class JudgeHyperparameters:
    def Is_all_nS_nonzero(self):
        if any(n_S == 0 for n_S in self.n_S):
            return False
        else:
            return True
        
    def Is_u_below_Lt_max(self, t, hypersamples, idx_reduced, Lt_max):
        xi, n_S, sum_y, beta, sigma = self.recall_parameters(t, idx_reduced)
        first = sum_y / n_S
        second = (beta @ sigma) / n_S
        third = (xi / n_S).sqrt()
        lhs = first + second + third
        print(f"index: {self.remaining_indices[idx_reduced]} Lt_max: {Lt_max:.2e}, 1st: {first:.2e} 2nd: {second:.2e} 3rd: {third:.2e}")
        if lhs < Lt_max:
            return True
        else:
            return False
    
    def recall_parameters(self, t, idx_reduced):
        idx = self.remaining_indices[idx_reduced]
        xi = self.compute_xi(t)
        n_S = self.n_S[idx]
        sum_y = sum([self.model.domain.transform_Y(y, update=False).item() for y in self.Y[idx]])
        beta = torch.tensor(self.beta[idx], dtype=self.dtype)
        sigma = torch.tensor(self.sigma[idx], dtype=self.dtype)
        return xi, n_S, sum_y, beta, sigma
    
    def Lt(self, t, idx_reduced):
        xi, n_S, sum_y, _, _ = self.recall_parameters(t, idx_reduced)
        Lt = sum_y / n_S - (xi / n_S).sqrt()
        return Lt.item()
    
    def max_Lt(self, t):
        Lt_max = max([
            self.Lt(t, idx_reduced) for idx_reduced in range(self.n_U)
        ])
        return Lt_max


class HyperparameterManager(HypersampleGenerator, RegretCalculator, MemoryManager, JudgeHyperparameters):
    def __init__(
        self, 
        model,
        g,                         # growth function
        regret_type="lengthscale", # regret type, ['lengthscale', ]
        delta=0.01,                # 0 < delta < 1. smaller delta leads to larger lower bound of lengthscale theta_L.
        C=1,                       # constant in lengthscale regret
        tol=1e-2,                  # tolerance paramter in xi
        R=1e-4,                    # noise variance (likelihood)
        kernel="RBF"
    ):
        model.g = g
        self.regret_type = regret_type
        self.delta = delta
        self.g = g
        self.C = C
        self.R = torch.tensor(R, dtype=model.dtype)
        self.tol = torch.tensor(tol, dtype=model.dtype)
        self.B = (model.domain.normed_bounds[1] - model.domain.normed_bounds[0]).prod().item()
        self.kernel = kernel
        self.ls_div = math.exp(1 / model.domain.bounds.shape[1]) if kernel == "RBF" else math.exp(1 / (model.domain.bounds.shape[1] + 2*2.5))
        
    def initialise_model(self, model):
        self.model = model
        self.gp = model.gp
        self.dtype = model.dtype
        self.n_dims = model.domain.bounds.shape[1]
        if self.regret_type == "lengthscale":
            self.idx_lengthscale = [i for i, name in enumerate(self.model.param_names) if "lengthscale" in name][0]
            self.constraint = self.gp.covar_module.base_kernel.raw_lengthscale_constraint
        
    def initialise_setup(self, model):
        self.initialise_model(model)
        hypersamples = self.generate_hypers()
        self.n_U = len(hypersamples)
        self.remaining_indices = list(range(self.n_U))
        self.initial_memory()
        return hypersamples
    
    def argmin_regret(self, hypersamples):
        regrets = torch.tensor([
            self.regret(hypersamples, index, shift=1) for index in range(self.n_U)
        ])
        idx_umin_reduced = regrets.argmin().item()
        return idx_umin_reduced
    
    def update_state(self, model, t, idx_umin_reduced, Y_next, beta, sigma):
        self.initialise_model(model)
        self.n_U = model.n_hypersamples
        self.update_memory(t, idx_umin_reduced, Y_next, beta, sigma)
    
    def update_hyperparameters(self, model, hypersamples, t, idx_umin_reduced, Y_next, beta, sigma):
        self.update_state(model, t, idx_umin_reduced, Y_next, beta, sigma)
        if self.Is_all_nS_nonzero():
            Lt_max = self.max_Lt(t)
            indices_accepted = []
            for idx_reduced in range(self.n_U-1, -1, -1):
                if self.Is_u_below_Lt_max(t, hypersamples, idx_reduced, Lt_max):
                    idx_rejected = self.remaining_indices[idx_reduced]
                    print(str(idx_rejected)+"-th hypersample is eliminated from candidates U.")
                else:
                    indices_accepted.append(idx_reduced)
            if not len(indices_accepted) == self.n_U:
                hypersamples= hypersamples[indices_accepted]
                self.remaining_indices = torch.tensor(self.remaining_indices)[indices_accepted].tolist()
                self.n_U = len(hypersamples)
            return hypersamples
        else:
            return hypersamples
        
    def add_new_hyperparameters(self, hypersamples, model, t):
        if self.theta_L / self.ls_div  > self.theta_H / self.g(t):
            print(f"New hypersample added with number: {len(self.S)}")
            new_ls = self.theta_L / self.ls_div 
            new_raw_ls = self.constraint.inverse_transform(new_ls)
            hypersamples = torch.cat([hypersamples, torch.tensor([[hypersamples[0][0], new_raw_ls]])])
            self.theta_L = new_ls
            self.remaining_indices = self.remaining_indices + [len(self.S)]
            self.n_U = len(hypersamples)
            self.expand_memory()

        return hypersamples