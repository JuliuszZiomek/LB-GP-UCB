import torch
import gpytorch
import botorch


def continuous_MLE(model):
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
    botorch.fit.fit_gpytorch_model(mll)
    model.gp.eval()
    return model

def continuos_berkenkamp(t, model, theta_LB=None):
    model = continuous_MLE(model)
    idx_lengthscale = [i for i, name in enumerate(model.param_names) if "lengthscale" in name][0]
    params = [param.item() for param_name, param in model.gp.named_parameters() if not "likelihood" in param_name]
    hyper_mle = torch.tensor(params, dtype=model.dtype)
    lengthscale_mle = model.gp.covar_module.base_kernel.raw_lengthscale_constraint.transform(hyper_mle[idx_lengthscale])
    
    gt = max(1, t**0.9)
    lengthscale_est = lengthscale_mle / gt
    
    if theta_LB is not None:
        if theta_LB > lengthscale_est:
            lengthscale_est = theta_LB
    
    raw_lengthscale = model.gp.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(lengthscale_est)
    hyper_mle[idx_lengthscale] = raw_lengthscale
    hyper_mle[0] = model.gp.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(1)).item()
    gp = model.set_model()
    model.gp = model.set_hypers(gp, hyper_mle)
    return model
