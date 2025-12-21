# -- import packages: ---------------------------------------------------------
import lightning
import torch

# -- callback cls: ------------------------------------------------------------
class StochasticWeightAveraging(lightning.pytorch.Callback):
    """
    Stochastic Weight Averaging callback.
    """
    def __init__(self, swa_lrs: float) -> None:
        """Initialize the callback.

        Args:
            swa_lrs (float): Learning rate for SWA.
        
        Returns:
            None
        """
        super().__init__()
        self.swa_lrs = swa_lrs
        self.swa_state = {}
        self.n_averaged = 0
    
    def on_fit_start(self, trainer, pl_module):
        # Initialize the dictionary to store parameter sums
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.swa_state[name] = torch.zeros_like(param.data)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Update the running average of parameters
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.swa_state:
                self.swa_state[name] += param.data
        self.n_averaged += 1
    
    def on_train_end(self, trainer, pl_module):
        # Transfer averaged parameters at the end of training
        if self.n_averaged > 0:
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.swa_state:
                    param.data.copy_(self.swa_state[name] / self.n_averaged)
            print(f"SWA completed with {self.n_averaged} models averaged.")