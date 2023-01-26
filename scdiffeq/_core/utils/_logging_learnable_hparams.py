
import torch

class LoggingLearnableHParams:
    def __init__(self, value, alt=None):
        
        self.value = value
        
        if not isinstance(alt, type(None)):
            setattr(self, "alt", alt)
        
    @property
    def is_int(self):
        return isinstance(self.value, int)
        
    @property
    def is_float(self):
        return isinstance(self.value, float)
    
    @property
    def is_tensor(self):
        return isinstance(self.value, torch.Tensor)
        
    @property
    def is_learnable_parameter(self):
        return isinstance(self.value, torch.nn.Parameter)
        
    
    def __call__(self):
        if (self.is_float) or (self.is_int):
            return self.value
        elif (self.is_learnable_parameter) or (self.is_tensor):
            if hasattr(self, "alt"):
                return self.alt
            return self.value.item()
