import copy
import torch
import gpytorch
import botorch


class HyperparameterManager:
    def check_hypers(self, gp):
        self.param_names = [param_name for param_name, param in gp.named_parameters() if not "likelihood" in param_name]
        self.n_hypers = len(self.param_names)
    
    def set_hypers(self, gp, hypersample):
        hypers = {param_name: param for param_name, param in zip(self.param_names, hypersample)}
        gp.initialize(**hypers).eval()
        return gp

