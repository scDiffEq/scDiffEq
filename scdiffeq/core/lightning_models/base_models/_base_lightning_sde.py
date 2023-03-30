
from ._base_lightning_diffeq import BaseLightningDiffEq

class BaseLightningSDE(BaseLightningDiffEq):
        
    def __init__(self, dt=0.1):
        super(BaseLightningSDE, self).__init__()
        from torchsde import sdeint
        self.sdeint = sdeint

    def integrate(self, X0, t, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
                
        return self.sdeint(self.func, X0=X0, ts=t, dt=self.hparams["dt"], **kwargs)