
from torchdiffeq import odeint
from torchsde import sdeint
import pydk
import torch

class ForwardFunctions:
    
    """
    Not sure if this is the best solution, but we can add the PRESCIENT forward
    functions to this as well.
    """
    
    def __init__(self, func, t_scale=0.02):
        
        self.t_scale = t_scale
        
        if func.mu and func.sigma:
            self.int = sdeint
            self.require_dt = True
        elif func.mu:
            self.int = odeint
            self.require_dt = False
        else:
            print("must define a drift func")

    def step(self, func, x0, t, **kwargs):
                
        if self.require_dt:
            return self.int(func, x0, t, kwargs['dt'])
        else:
            device = torch.get_device(x0)
            _t = (pydk.min_max_normalize(t)*self.t_scale).to(device)
            return self.int(func, x0, _t)