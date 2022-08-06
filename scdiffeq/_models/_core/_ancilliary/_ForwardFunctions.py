
from torchdiffeq import odeint
from torchsde import sdeint
import pydk
import torch

from ._manual_forward_step import _manual_forward_step

class ForwardFunctions:
    
    """
    Not sure if this is the best solution, but we can add the PRESCIENT forward
    functions to this as well.
    """
    
    def __init__(self, func, t_scale=0.02):
        
        self.t_scale = t_scale
        self._neural_net = False
        self._neural_diffeq = False
        
        if hasattr(func, "mu") and hasattr(func, "sigma"):
            self._neural_diffeq = True
            self.int = sdeint
            self.require_dt = True
        elif hasattr(func, "mu"):
            self._neural_diffeq = True
            self.int = odeint
            self.require_dt = False
        else:
            self._neural_net = True
            # using a non-neural-diffeq (i.e., regular forward_net or potential_net)
            self.int = _manual_forward_step
            

    def step(self, func, x0, t, dt, stdev, tspan, device):
                
        if self._neural_diffeq:
            if self.require_dt:
                kwargs = {"dt":dt}
                return self.int(func, x0, t, **kwargs).to(device)
            else:
                device = torch.get_device(x0)
                _t = (pydk.min_max_normalize(t)*self.t_scale).to(device)
                return self.int(func, x0, _t)

        if self._neural_net:
            timestep_size = 2 # make flexible later
            n_timesteps = int(tspan / timestep_size)
            x_step = x0
            x_forward = [x0]
            for step in range(n_timesteps):
                x_step = _manual_forward_step(func, x_step, dt, stdev, tspan, device)
                x_forward.append(x_step)
                
            return torch.stack(x_forward)
        

# class ForwardFunctions:
    
#     """
#     Not sure if this is the best solution, but we can add the PRESCIENT forward
#     functions to this as well.
#     """
    
#     def __init__(self, func, t_scale=0.02):
        
#         self.t_scale = t_scale
        
#         if func.mu and func.sigma:
#             self.int = sdeint
#             self.require_dt = True
#         elif func.mu:
#             self.int = odeint
#             self.require_dt = False
#         else:
#             print("must define a drift func")

#     def step(self, func, x0, t, kwargs):
                
#         if self.require_dt:
#             return self.int(func, x0, t, **kwargs)
#         else:
#             device = torch.get_device(x0)
#             _t = (pydk.min_max_normalize(t)*self.t_scale).to(device)
#             return self.int(func, x0, _t)