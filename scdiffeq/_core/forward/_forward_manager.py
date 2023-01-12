
# -- import packages: ------------------------------------------------------------------
import torch
import numpy as np
from autodevice import AutoDevice


# -- import local dependencies: --------------------------------------------------------
from ..utils import Base, extract_func_kwargs
from ._function_credentials import Credentials
from ._potential_regularizer import PotentialRegularizer
from ._batch import Batch


# -- supporting classes / functions: ---------------------------------------------------
class UniversalForwardIntegrator(Base):
    def __init__(self, func, adjoint=False):
        super(Base, self).__init__()

        self.func = func
        creds = Credentials(self.func, adjoint=adjoint)
        self.func_type, self.mu_is_potential, self.sigma_is_potential = creds()
        self.integrator = creds.integrator

    def __call__(
        self,
        X0,
        t,
        dt=0.1,
        stdev=0.5,
        device=AutoDevice(),
        max_steps=None,
        return_all=False,
    ):

        self.__parse__(locals(), ignore=["self", "X0", "func", "t"])
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self._KWARGS)
        # for some reason, locals(), which gets passed on to self._KWARGS as part of the
        # Base class, is carrying all arguments from the main model class. For now, I need
        # manually remove / pop these.
        if "func" in int_kwargs.keys():
            func = int_kwargs.pop("func")
        if "t" in int_kwargs.keys():
            t = int_kwargs.pop("t").to(device)
        return self.integrator(self.func, X0, t, **int_kwargs)

# -- main class: -----------------------------------------------------------------------

class ForwardManager(Base):
    """passed and executed during Lightningmodel.forward()"""

    def __init__(
        self,
        model,
        tau=1e-06,
        burn_steps=100,
        t=None,
        expand=False,
    ):
        """passing t here is reserved for special scenarios... not used regularly"""
        self.__config__(locals())

    def __config__(self, kwargs, private=["t"], ignore=["self"]):

        self.__parse__(kwargs, private=private, ignore=ignore)

    def _configure_t(self):
        if isinstance(self._t, NoneType):
            self._t = self.batch.t

    @property
    def forward_func(self):
        return UniversalForwardIntegrator(
            func=self.model.func, adjoint=self.model.adjoint
        )

    @property
    def func_type(self):
        return self.forward_func.func_type

    @property
    def t(self):
        self._configure_t()
        return self._t

    @property
    def batch(self):
        return Batch(
            batch=self._batch,
            stage=self.stage,
            func_type=self.func_type,
            expand=self.expand,
        )

    @property
    def potential_regularizer(self):
        return PotentialRegularizer(
            self.batch,
            self.model,
            tau=self.tau,
            burn_steps=self.burn_steps,
        )

    @property
    def do_regularize(self):
        return (self.tau > 0) and (self.model.mu_is_potential)

    @property
    def use_velocity(self):
        return True

    @property
    def use_fate_bias(self):
        return True

    @property
    def n_fates(self):
        return self.model.adata.uns["fate_df"].shape[1]

    @property
    def input_dim(self):
        return self.batch.X.shape[-1]

    @property
    def fated_mask(self):
        """
        we can take the first (0th) set in the index bc we only care about d2 for fate
        """
        return [
            str(cell) in self.model.adata.uns["fated_cell_idx"]
            for cell in self.batch.cell_idx[0].flatten()
        ]

    @property
    def batch_idx_fated(self):
        return self.batch.cell_idx[0].flatten()[self.fated_mask]

    @property
    def F(self):
        return torch.Tensor(
            self.model.adata.uns["fate_df"].loc[self.batch_idx_fated].values
        ).to(AutoDevice())

    @property
    def Linear(self):
        return torch.nn.Linear(in_features=self.input_dim, out_features=self.n_fates).to(self.model.device)

    # -- methods: ----------------------------------------------------
    def calculate_velocity(self, X_hat):
        return torch.diff(X_hat, n=1, dim=0, append=X_hat[-1:, :, :])

    def __call__(self, batch, batch_idx=0, stage="fit", stdev=0.5):

        self.__parse__(locals(), private=["batch"])

        FORWARD_OUTS = {}

        # -- (1) run forward inference: ------------------------------
        FORWARD_OUTS["X"] = self.batch.X
        FORWARD_OUTS["X_hat"] = self.forward_func(
            X0=self.batch.X0,
            t=self.batch.t,
            dt=self.model.dt,
            stdev=self.stdev,
            device=self.model.device,
        )

        # -- (2) run regularizer: -------------------------------------
        if self.do_regularize:
            (
                FORWARD_OUTS["ref_psi"],
                FORWARD_OUTS["burn_psi"],
            ) = self.potential_regularizer(self.forward_func, self.stdev)
            FORWARD_OUTS["tau"] = self.tau

        # -- (3) calculate velocity: ----------------------------------
        if self.use_velocity:
            FORWARD_OUTS["V"] = self.batch.V
            FORWARD_OUTS["V_hat"] = self.calculate_velocity(FORWARD_OUTS["X_hat"])

        # -- (4) calculate fate bias: ----------------------------------
        if self.use_fate_bias:
            FORWARD_OUTS["F"] = self.F
            FORWARD_OUTS["F_hat"] = self.Linear(
                FORWARD_OUTS["X_hat"][1:][0, self.fated_mask, :].to(self.model.device)
            )

        # -- paperwork: ------------------------------------------------
        FORWARD_OUTS["model"] = self.model
        FORWARD_OUTS["stage"] = self.batch.stage

        return FORWARD_OUTS
