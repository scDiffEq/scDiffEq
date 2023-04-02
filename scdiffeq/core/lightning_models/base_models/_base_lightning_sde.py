
from ._base_lightning_diffeq import BaseLightningDiffEq

class BaseLightningSDE(BaseLightningDiffEq):
        
    def __init__(self):
        super(BaseLightningSDE, self).__init__()
        from torchsde import sdeint
        self.sdeint = sdeint

    def integrate(self, X0, t, dt, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
                
        return self.sdeint(sde=self.func, y0=X0, ts=t, dt=dt, **kwargs)