import copy
import pyro
import torch
import gpytorch
import botorch
from abc import ABC, abstractmethod
from ._hyperparameters import HyperparameterManager
from ._acquisition_function import NormalUCB, ExpectedUCB, UUCB
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood

class TemplateGP(ABC):
    @abstractmethod
    def train(self, hypersamples, verbose=False):
        r"""Training the hyperparamters of the GP model"""
        pass

class BaseGP(TemplateGP, HyperparameterManager):
    def __init__(self, train_X, train_Y, domain, mean_module=None, covar_module=None):
        self.domain = domain
        if mean_module is None:
            mean_module = gpytorch.means.ZeroMean()
        self.mean_module = copy.deepcopy(mean_module)
        
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=None),
            )
        self.covar_module = copy.deepcopy(covar_module)
        self.conditioning(train_X, train_Y)
        self.initialisation()
        self.dtype = train_Y.dtype
        self.num_outputs = 1
        self.is_fully_bayesian = False
    
    def initialisation(self):
        self.gp = self.set_model()
        self.check_hypers(self.gp)
        
    def conditioning(self, train_X, train_Y):
        self.train_X_norm = self.domain.transform_X(train_X)
        self.train_Y_norm = self.domain.transform_Y(train_Y, update=True)
    
    def set_model(self):
        gp = botorch.models.SingleTaskGP(
            self.train_X_norm,
            self.train_Y_norm,
            mean_module = copy.deepcopy(self.mean_module),
            covar_module = copy.deepcopy(self.covar_module),
            likelihood=GaussianLikelihood()
        )
        # We fix the likelihood variance
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(1e-4),
            'covar_module.outputscale': torch.tensor(1),
        }
        gp.initialize(**hypers)
        gp.likelihood.raw_noise.requires_grad = False
        gp.covar_module.raw_outputscale.requires_grad = False
        return gp
        
    def set_batch_gps(self, hypersamples):
        self.n_hypersamples = len(hypersamples)
        models_with_different_hypers = [
            self.set_hypers(self.set_model(), hypersample) for hypersample in hypersamples
        ]
        gps = gpytorch.models.IndependentModelList(
            *models_with_different_hypers
        ).eval()
        likelihoods = gpytorch.likelihoods.LikelihoodList(
            *[model.likelihood for model in models_with_different_hypers]
        )
        return gps, likelihoods
    
    def predictive_mean_and_stddev(self, X):
        if len(X.shape) == 3:
            X = X.squeeze(1)
        pred = self.gp.likelihood(self.gp(X))
        return pred.mean, pred.stddev
    
    def batch_predictive_mean_and_stddev(self, X):
        if len(X.shape) == 3:
            X = X.squeeze()
        batch_pred = self.likelihoods(*self.gps(*[X for _ in range(self.n_hypersamples)]))
        pred_means = torch.vstack([pred.mean for pred in batch_pred])
        pred_stddevs = torch.vstack([pred.stddev for pred in batch_pred])
        return pred_means, pred_stddevs
    

class GPEnsemble(BaseGP):
    def __init__(self, train_X, train_Y, domain, method="UUCB", mean_module=None, covar_module=None):
        super().__init__(train_X, train_Y, domain, mean_module=mean_module, covar_module=covar_module)
        if not method in ['ExpectedUCB', 'UUCB']:
            raise NotImplementedError("The method is either 'expectedUCB', or 'UUCB'.")
        self.method = method
        
    def train(self, hypersamples, verbose=False):
        self.gps, self.likelihoods = self.set_batch_gps(hypersamples)
        

class SingleGP(BaseGP):
    def __init__(self, train_X, train_Y, domain, method="MLE", mean_module=None, covar_module=None):
        super().__init__(train_X, train_Y, domain, mean_module=mean_module, covar_module=covar_module)
        if not method in ["MLE", "random", "berkenkamp", "LB-GP-UCB", "cMLE", "cBerkenkamp", "cBerkenkamp_LB", "oracle"]:
            raise NotImplementedError("The method is either 'random', 'MLE', 'berkenkamp', or 'HB-GP-UCB' or 'oracle'.")
        self.method = method
        
    def train(self, hypersamples, t=None, verbose=False, g=None, theta_0=None, oracle_theta=None):
        self.gp = self.set_model()
        self.n_hypersamples = len(hypersamples)
        if self.method == "random":
            hypers = self.random_sample(hypersamples, verbose=verbose)
            self.gp = self.set_hypers(self.gp, hypers)
        elif self.method == "MLE":
            hypers = self.discrete_mle(hypersamples, verbose=verbose)
            self.gp = self.set_hypers(self.gp, hypers)
        elif self.method == "berkenkamp":
            if t is None:
                raise ValueError("t should be given as input")
            hypers = self.berkenkamp(t, hypersamples, verbose=verbose)
            self.gp = self.set_hypers(self.gp, hypers)
        elif self.method == "cMLE" or self.method=="turbo":
            self.continuous_MLE()
            return
        elif self.method == "cBerkenkamp":
            if t is None or g is None:
                raise ValueError("t and g should be given as input")
            hypers = self.continuos_berkenkamp(t, g, theta_0=theta_0)
            self.gp = self.set_hypers(self.gp, hypers)
        elif self.method == "oracle":
            if oracle_theta is None:
                raise ValueError("oracle_theta should be given as input")
            oracle_hypers = self.get_oracle_hypers(torch.tensor(oracle_theta).reshape(1, 1))
            self.gp = self.set_hypers(self.gp, oracle_hypers)
        
    def set_hyper(self, hyper, hypersamples):
        self.n_hypersamples = len(hypersamples)
        gp = self.set_model()
        self.gp = self.set_hypers(gp, hyper)
    
    def random_sample(self, hypersamples, verbose=False):
        idx = torch.randint(len(hypersamples), (1,)).item()
        
        if verbose:
            print('%d-th hypersample is selected' % (idx))
        hypers_random = hypersamples[idx]
        return hypers_random
    
    def discrete_mle(self, hypersamples, verbose=False, index=False):
        idx_min = 0
        loss_best = 1e100
        
        gps, likelihoods = self.set_batch_gps(hypersamples)
        for idx, (_likelihood, _model) in enumerate(zip(likelihoods.likelihoods, gps.models)):
            _model.train()
            _likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likelihood, _model)
            output = _model(self.train_X_norm)
            loss = -mll(output, self.train_Y_norm.squeeze()).item()
            _model.eval()
            _likelihood.eval()
            if loss < loss_best:
                idx_min = idx
                loss_best = loss
        
        if verbose:
            print('%d-th hypersample is selected' % (idx_min))
        
        if index:
            return idx_min
        else:
            return hypersamples[idx_min]
        
    def berkenkamp(self, t, hypersamples, verbose=False, index=False):
        idx_min = self.discrete_mle(hypersamples, verbose=False, index=True)
        idx_lengthscale = [i for i, name in enumerate(self.param_names) if "lengthscale" in name][0]
        lengthscales = self.gp.covar_module.base_kernel.raw_lengthscale_constraint.transform(hypersamples[:, idx_lengthscale])
        
        lengthscale_map = lengthscales[idx_min]
        gt = max(1, t**0.9)
        lengthscale_est = lengthscale_map / gt
        idx_selected = (lengthscales - lengthscale_est).pow(2).argmin().item()
        if verbose:
            print('%d-th hypersample is selected' % (idx_selected))
        
        if index:
            return idx_selected
        else:
            return hypersamples[idx_selected]
        
    def continuous_MLE(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        botorch.fit.fit_gpytorch_model(mll)
        self.gp.eval()

    def continuos_berkenkamp(self, t, g, theta_LB=None, theta_0=None):
        idx_lengthscale = [i for i, name in enumerate(self.param_names) if "lengthscale" in name][0]
        params = [param.item() for param_name, param in self.gp.named_parameters() if not "likelihood" in param_name]
        hyper_mle = torch.tensor(params, dtype=self.dtype)
        
        gt = g(t)
        lengthscale_est = theta_0 / gt
        
        if theta_LB is not None:
            if theta_LB > lengthscale_est:
                lengthscale_est = theta_LB
        
        raw_lengthscale = self.gp.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(lengthscale_est)
        hyper_mle[idx_lengthscale] = raw_lengthscale
        hyper_mle[0] = self.gp.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(1)).item()
        print(f"Using length scale value: {lengthscale_est}")
        return hyper_mle
    
    def get_oracle_hypers(self, oracle_theta):
        idx_lengthscale = [i for i, name in enumerate(self.param_names) if "lengthscale" in name][0]
        params = [param.item() for param_name, param in self.gp.named_parameters() if not "likelihood" in param_name]
        hyper_mle = torch.tensor(params, dtype=self.dtype)
        
        raw_lengthscale = self.gp.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(oracle_theta)
        hyper_mle[idx_lengthscale] = raw_lengthscale
        hyper_mle[0] = self.gp.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(1)).item()
        return hyper_mle


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, covar_module, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FullyBayesianGP(BaseGP):
    def __init__(self, train_X, train_Y, domain, method="FullyBayesianGP", mean_module=None, covar_module=None):
        super().__init__(train_X, train_Y, domain, mean_module=mean_module, covar_module=covar_module)
        self.reset_gp()
        self.method = method
        self.previous_mcmc_samples = None
    
    def set_model(self):
        gp = ExactGPModel(
            self.train_X_norm,
            self.train_Y_norm,
            mean_module = copy.deepcopy(self.mean_module),
            covar_module = copy.deepcopy(self.covar_module),
            likelihood=GaussianLikelihood()
        )
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.05),
            'covar_module.outputscale': torch.tensor(1),
        }
        gp.initialize(**hypers)
        gp.likelihood.raw_noise.requires_grad = False
        gp.covar_module.raw_outputscale.requires_grad = False
        return gp

    def reset_gp(self):
        self.gp = self.set_model()
        self.gp.covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(3.0, 6.0), "lengthscale")
            
    def train(self, n_mcmc=100, burnin=100):
        self.reset_gp()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        mcmc_samples = None
        while mcmc_samples is None:
            try:
                def pyro_model(x, y):
                    with gpytorch.settings.fast_computations(False, False, False):
                        sampled_model = self.gp.pyro_sample_from_prior()
                        output = sampled_model.likelihood(sampled_model(x))
                        pyro.sample("obs", output, obs=y)
                    return y
                
                nuts_kernel = pyro.infer.mcmc.NUTS(pyro_model)
                mcmc_run = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=n_mcmc, warmup_steps=burnin)
                mcmc_run.run(self.train_X_norm, self.train_Y_norm)
                mcmc_samples = mcmc_run.get_samples()
            except Exception as e:
                mcmc_samples = self.previous_mcmc_samples

        self.gp.pyro_load_from_samples(mcmc_samples)
        self.gp.eval()
        self.previous_mcmc_samples = mcmc_samples
        self.n_hypersamples = n_mcmc

    def conditioning(self, train_X, train_Y):
        self.train_X_norm = self.domain.transform_X(train_X)
        self.train_Y_norm = self.domain.transform_Y(train_Y, update=True).reshape(-1)
        
    def predictive_mean_and_stddev(self, X):
        if len(X.shape) == 3:
            X = X.squeeze(1)
        pred = self.gp.likelihood(self.gp(X))
        mean = pred.mean.mean(axis=0)
        stddev = pred.variance.mean(axis=0)
        return mean, stddev
    