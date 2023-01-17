

# -- import packages: --------------------------------------------------------------------
from autodevice import AutoDevice
import numpy as np
import torch



# -- import local dependencies: ----------------------------------------------------------
from ..utils import Base, extract_func_kwargs
from ._function_credentials import Credentials
from ._potential_regularizer import PotentialRegularizer
from ._batch import Batch


# -- typing: -----------------------------------------------------------------------------
NoneType = type(None)


# -- supporting classes / functions: -----------------------------------------------------
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
        fate_scale=0,
        return_all=False,
    ):
        """
        Notes:
        ------
        (1) For some reason, locals(), which gets passed on to  self._KWARGS as part of  the
            Base class, is carrying all arguments from the main model class. For now, I need
            to manually remove / pop these.
        """
        self.__parse__(locals(), ignore=["self", "X0", "func", "t"])
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self._KWARGS)

        if "func" in int_kwargs.keys():
            func = int_kwargs.pop("func")
        if "t" in int_kwargs.keys():
            t = int_kwargs.pop("t").to(device)
        return self.integrator(self.func, X0, t, **int_kwargs)


class ForwardManager(Base):
    """passed and executed during Lightningmodel.forward()"""

    def __init__(
        self,
        model,
        tau=1e-06,
        fate_scale=0,
        burn_steps=100,
        t=None,
        expand=False,
        velo_gene_idx=None,
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
    def potential_switch(self):
        return ((self.tau > 0) and (self.model.mu_is_potential) and (not self.model.hparams['disable_potential']))

    @property
    def velocity_switch(self):
        return ((not isinstance(self.velo_gene_idx, NoneType)) and (not self.model.hparams['disable_velocity']))

    @property
    def fate_bias_switch(self):
        return ((self.fate_scale > 0) and (not self.model.hparams['disable_fate_bias']))

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

    def _config_pca(self):
        if not hasattr(self, "_pca"):
            self._pca = self.model.adata.uns['pca']
    @property
    def pca(self):
        self._config_pca()
        return self._pca

    # -- methods: ----------------------------------------------------
    def calculate_velocity(self, X_hat):
        
        X_hat_np = X_hat.detach().cpu().numpy()
        X_hat_ = torch.stack([torch.Tensor(self.pca.inverse_transform(xt)) for xt in X_hat_np]).to(AutoDevice())
        X_hat_ = X_hat_[:, :, self.velo_gene_idx]
        return torch.diff(X_hat_, n=1, dim=0, append=X_hat_[-1][None, :, :])

    def __call__(self, batch, batch_idx=0, stage="fit", stdev=0.5):

        self.__parse__(locals(), private=["batch"])

        FORWARD_OUTS = {}

        # -- (1) run forward inference: --------------------------------------------------
        FORWARD_OUTS["X"] = self.batch.X
        FORWARD_OUTS["X_hat"] = self.forward_func(
            X0=self.batch.X0,
            t=self.batch.t,
            dt=self.model.hparams['dt'],
            stdev=stdev,
            device=self.model.device,
        )

        # -- (2) run regularizer: ------------------------------------------------------
        if self.potential_switch:
            (
                FORWARD_OUTS["ref_psi"],
                FORWARD_OUTS["burn_psi"],
            ) = self.potential_regularizer(self.forward_func, self.stdev)

        # -- (3) calculate velocity: ---------------------------------------------------
        if self.velocity_switch:
            # add linear transform for PC -> gene space to calculate velo.
            FORWARD_OUTS["V"] = self.batch.V[:, :, self.velo_gene_idx]
            FORWARD_OUTS["V_hat"] = self.calculate_velocity(FORWARD_OUTS["X_hat"])

        # -- (4) calculate fate bias: --------------------------------------------------
        if self.fate_bias_switch:
            FORWARD_OUTS["F"] = self.F
            FORWARD_OUTS["F_hat"] = self.Linear(
                FORWARD_OUTS["X_hat"][1:][0, self.fated_mask, :].to(self.model.device)
            )
            
        return FORWARD_OUTS
