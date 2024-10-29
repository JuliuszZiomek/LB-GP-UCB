import random
import torch
import botorch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._information_gain import InformationGain
from ._domain import FiniteDomain


class NormalUCB(AnalyticAcquisitionFunction):
    def __init__(self, model, beta):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        self.register_buffer("beta", beta)
        
    @botorch.utils.t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        pred_mean, pred_std = self.model.predictive_mean_and_stddev(X)
        return pred_mean + self.beta * pred_std


class BetaCalculator(InformationGain):
    def __init__(self, model, n_iterations, tol=1e-2, gamma=20, R=None, method="exact"):
        super().__init__(model.gp, n_iterations, model.dtype, gamma=gamma)
        self.update_model(model)
        self.method = method
        self.tol = torch.tensor(tol, dtype=model.dtype)
        self.B = (model.domain.normed_bounds[1] - model.domain.normed_bounds[0]).prod()
        if not R is None:
            self.R = torch.tensor(R, dtype=model.dtype)
        
    def update_model(self, model):
        self.model = model
        self.dtype = model.dtype
        n_U = model.n_hypersamples
        
    def compute_beta(self, t, gp):
        if self.method == "bayesian":
            d = gp.train_inputs[0].shape[1]
            beta = (2 * d * torch.log(t**2 / self.tol)).sqrt()
        else:
            if hasattr(self, "R"):
                R = self.R
            else:
                R = gp.likelihood.noise.detach()
            
            try:
                information_gain = self.compute_information_gain(t, gp, method=self.method)
                beta = self.B + R * (
                    2 * (information_gain + 1 + (2 / self.tol).log())
                ).sqrt()
                
                if torch.isnan(beta) or torch.isinf(beta) or (beta < 0):
                    print("Beta computation fails, we set beta as 1.")
                    print(beta)
                    beta = torch.tensor(1, dtype=self.dtype)
            except:
                print("Beta computation fails, we set beta as 1.")
                beta = torch.tensor(1, dtype=self.dtype)
        return beta
    
    def batch_beta(self, t):
        return torch.tensor([
            self.compute_beta(t, gp) for gp in self.model.gps.models
        ])


class AcquisitionFunction(BetaCalculator):
    def __init__(self, model, n_iterations, tol=1e-2, gamma=20, R=None, method="exact", bounds=None):
        super().__init__(model, n_iterations, tol=tol, gamma=gamma, R=R, method=method)
        if bounds is not None:
            bounds[0,:] = torch.max(bounds[0,:], model.domain.normed_bounds[0,:])
            bounds[1,:] = torch.min(bounds[1,:], model.domain.normed_bounds[1,:])
            self.bounds = bounds
        else:
            self.bounds = model.domain.normed_bounds
        
    def acquisition_function(self, beta):
        acqf = NormalUCB(self.model, beta)
        return acqf
    
    def calc_beta(self, t):
        if self.model.method == "FullyBayesianGP":
            beta = torch.tensor(1, dtype=self.dtype)
        else:
            beta = self.compute_beta(t, self.model.gp)
        return beta
    
    def optimize(self, t):
        self.beta = self.calc_beta(t)
        acqf = self.acquisition_function(self.beta)
        if type(self.model.domain) == FiniteDomain:
            normed_dataset = self.model.domain.normed_dataset
            available_ix = torch.all((normed_dataset >= self.bounds[0,:]) & (normed_dataset <= self.bounds[1,:]), dim=1)
            normed_available_points = normed_dataset[available_ix]
            available_points = self.model.domain.dataset[available_ix]

            if available_points.numel() == 0:
                print("Warning, current TR contains no points. Selecting point randomly.")
                return available_points[random.choice(range(available_points.shape[0]))]

            return available_points[torch.argmax(acqf(normed_available_points.unsqueeze(1))), :].unsqueeze(0)
        else:
            X_next, _ = botorch.optim.optimize_acqf(
                acqf,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            return self.model.domain.untransform_X(X_next)
