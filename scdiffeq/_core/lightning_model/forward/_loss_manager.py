
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """TODO"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
import numpy as np
import torch


# -- import local dependencies: ----------------------------------------------------------
from ...utils import sum_normalize, AutoParseBase
from ._sinkhorn_divergence import SinkhornDivergence
from ._loss_logger import LossLogger


# -- typing: -----------------------------------------------------------------------------
NoneType = type(None)



# -- loss manager: -----------------------------------------------------------------------
class LossManager(AutoParseBase):
    """Catch the outputs of the forward function and perform corresponding computations."""
    LossDict = {}
    
    def __init__(self, model, stage):
        
        self.__config__(locals())
    
    def __assume_hparams__(self):
        
        self.hparams = self.model.hparams
        
        for key, val in self.hparams.items():
            setattr(self, key, val)
            
    def __config__(self, kwargs):
        
        self.SinkDiv = SinkhornDivergence()
        self.MSE = torch.nn.MSELoss()

        self.__parse__(kwargs)
        self.__assume_hparams__()

    # -- helpers: ------------------------------------------------------------------------
    def _format(self, X_arr):
        if not self.real_time:
            return X_arr[1:].flatten(0, 1)[self.shuffle_idx][None, :, :].contiguous()
        return X_arr.contiguous()

    def _format_weight(self, W_arr, X_like):
        if not isinstance(W_arr, torch.Tensor):
            return sum_normalize(torch.ones_like(X_like)[:, :, 0])[:, :, None]

    @property
    def n_cells(self):
        return self._X.shape[1]

    @property
    def shuffle_idx(self):
        return np.random.choice(range(self.n_cells), size=self.n_cells, replace=False)

    
    # -- key batch properties: -----------------------------------------------------------
    @property
    def X(self):
        return self._format(self._X)

    @property
    def X_hat(self):
        return self._format(self._X_hat)

    @property
    def W(self):
        return self._format_weight(self._W, self.X)

    @property
    def W_hat(self):
        return self._format_weight(self._W_hat, self.X_hat)

    @property
    def V(self):
        if not isinstance(self._V, NoneType):
            return self._format(self._V) * self.V_coefficient * self.V_scaling

    @property
    def V_hat(self):
        if not isinstance(self._V_hat, NoneType):
            return self._format(self._V_hat) * self.V_coefficient

    @property
    def F(self):
        if not isinstance(self._F, NoneType):
            return self._F

    @property
    def F_hat(self):
        if not isinstance(self._F_hat, NoneType):
            return self._F_hat

    # -- requisite enabling properties for each type of loss: ----------------------------
    @property
    def potential_enabled(self):
        return (self.tau > 0)
    
    @property
    def velocity_enabled(self):
        return (
            (not isinstance(self.V, NoneType)) and
            (not isinstance(self.V_hat, NoneType))
        )

    @property
    def fate_bias_enabled(self):
        return (
            (not isinstance(self.F, NoneType)) and
            (not isinstance(self.F_hat, NoneType))
        )

    # -- switches: -----------------------------------------------------------------------
    @property
    def velocity_switch(self):
        return (self.velocity_enabled) and (not self.disable_velocity)
    
    @property
    def potential_switch(self):
        return (self.potential_enabled) and (not self.disable_potential)
    
    @property
    def fate_bias_switch(self):
        return (self.fate_bias_enabled) and (not self.disable_fate_bias)

    # -- loss computations: --------------------------------------------------------------
    def compute_positional_loss(self):
        """Always compute positional loss"""
#         print("W, X, W_hat, X_hat", self.W, self.X, self.W_hat, self.X_hat)
#         print("W, X, W_hat, X_hat", self.W.shape, self.X.shape, self.W_hat.shape, self.X_hat.shape)
        self.LossDict["positional"] = self.SinkDiv(self.W,
                                                   self.X,
                                                   self.W_hat,
                                                   self.X_hat,
                                                  )
        
        
    def compute_positional_velocity_loss(self):
        """TODO - velocity-specific weights"""
        if self.velocity_switch:
            self.LossDict["positional_velocity"] = self.SinkDiv(
                self.W,
                torch.concat([self.X, self.V], axis=-1),
                self.W_hat,
                torch.concat([self.X_hat, self.V_hat], axis=-1),
            )
            
    def compute_potential_loss(self):
        if self.potential_switch:
            self.LossDict["potential"] = ((self._ref_psi + self._burn_psi) * self.tau).abs()

    def compute_fate_bias_loss(self):
        if self.fate_bias_switch:
            self.LossDict["fate_bias"] = self.MSE(self.F, self.F_hat) * self.fate_scale
    
    # -- logging and formatting: ---------------------------------------------------------
    @property
    def backprop_losses(self):
        _backprop_losses = []
        for key in self.__dir__():
            if key.startswith("skip_") and (not getattr(self, key)):
                _backprop_losses.append(key.split("skip_")[1].split("_backprop")[0])                    
        return _backprop_losses
    
    def log_loss_values(self):
        logger = LossLogger(backprop_losses=self.backprop_losses)
        self.BackPropDict = logger(
            model = self.model,
            LossDict = self.LossDict,
            stage = self.stage,
        )

    def format_for_backprop(self):
#         for k, v in self.BackPropDict.items():
#             print("{} | {}".format(k, v))
        return torch.hstack(list(self.BackPropDict.values())).sum()

    # -- run all: ------------------------------------------------------------------------
    def __call__(
        self,
        X,
        X_hat,
        W=None,
        W_hat=None,
        V=None,
        V_hat=None,
        F=None,
        F_hat=None,
        ref_psi=None,
        burn_psi=None,
    ):

        """
        Notes:
        ------
        (1) psi loss is already computed by the PotentialRegularizer, so we pass it here
            to bring all losses together.
        (2) For velo, potential, and fate bias loss there are two flags to consider: is
            it (a) enabled? If so, always compute, unless specifically disabled. Should
            the value (b) be backprop'd over? If computed, always log.
        """

        self.__parse__(locals(), public=[None])

        # -- (1) compute positional loss: ------------------------------------------------
        self.compute_positional_loss()
        
        # -- (2) compute auxiliary loss: -------------------------------------------------
        self.compute_positional_velocity_loss()
        self.compute_potential_loss()
        self.compute_fate_bias_loss()
        
        # -- (3) log computed loss vals: -------------------------------------------------
        self.log_loss_values()
        
        # -- (4) format and return for backprop: -----------------------------------------
        return self.format_for_backprop()
