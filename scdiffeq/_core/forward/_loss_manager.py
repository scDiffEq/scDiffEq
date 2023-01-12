
import torch
from ..utils import Base, sum_normalize
from ._sinkhorn_divergence import SinkhornDivergence
NoneType = type(None)

class LossLog(Base):
    def __init__(self, unlogged_stages=["predict", "test"]):
        self.__parse__(locals())

    def __call__(self, model, LossDict, stage):

        for loss_type, log_vals in LossDict.items():
            if log_vals.dim() == 0:
                log_msg = "{}_{}_{}".format(stage, 0, loss_type)
                log_val = log_vals
                model.log(log_msg, log_val)
            else:
                for i, val in enumerate(log_vals):
                    log_msg = "{}_{}_{}".format(stage, i, loss_type)
                    log_val = val
                    model.log(log_msg, log_val)


class LossManager(Base):
    """Catch the outputs of the forward function and perform corresponding computations."""

    LossDict = {}

    def __init__(self, real_time):

        sinkhorn_divergence = SinkhornDivergence()
        mse = torch.nn.MSELoss()

        self.__parse__(locals())

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

    # -- key properties: -------
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
            return self._format(self._V) * self._velo_coef * self._velo_scale

    @property
    def V_hat(self):
        if not isinstance(self._V_hat, NoneType):
            return self._format(self._V_hat) * self._velo_coef

    @property
    def F(self):
        if not isinstance(self._F, NoneType):
            return self._F

    @property
    def F_hat(self):
        if not isinstance(self._F_hat, NoneType):
            return self._F_hat

    @property
    def use_velocity(self):
        return (not isinstance(self.V, NoneType)) and (
            not isinstance(self.V_hat, NoneType)
        )

    @property
    def use_fate_bias(self):
        return (not isinstance(self.F, NoneType)) and (
            not isinstance(self.F_hat, NoneType)
        )

    # -- calculations: ----------
    def psi(self):
        if (not isinstance(self._ref_psi, NoneType)) and (
            not isinstance(self._burn_psi, NoneType)
        ):
            return ((self._ref_psi + self._burn_psi) * self._tau).abs()

    def positional(self):
        return self.sinkhorn_divergence(self.W, self.X, self.W_hat, self.X_hat)

    def positional_velocity(self):

        W = torch.concat([self.W, self.W], axis=1) # TODO - velocity-specific weights
        X = torch.concat([self.X, self.V], axis=1)
        W_hat = torch.concat([self.W_hat, self.W_hat], axis=1)
        X_hat = torch.concat([self.X_hat, self.V_hat], axis=1)

        return self.sinkhorn_divergence(W, X, W_hat, X_hat)

    def fate_bias(self):
        return self.mse(self.F, self.F_hat) * self._fate_scale

    def __total__(self):
        return torch.hstack(list(self.LossDict.values())).sum()

    # -- run all: ----------------
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
        tau=1e-06,
        fate_scale=100,
        velo_coef=1,  # provided
        velo_scale=1,  # learned
        model=None,
        stage="fit",
    ):

        """
        Notes:
        ------
        (1) psi loss is already computed by the PotentialRegularizer, so we pass it here to bring
            all losses together.
        """

        self.__parse__(locals(), public=[None])

        # -- (5) compute losses: ---------------------------------------
        if self.use_velocity:
            self.LossDict["positional_velocity"] = pv = self.positional_velocity()
        else:
            self.LossDict["positional"] = p = self.positional()

        self.LossDict["psi"] = psi = self.psi()

        if self.use_fate_bias:
            self.LossDict["fate_bias"]= fb = self.fate_bias()

        # -- (6) log losses: -------------------------------------------
        if not isinstance(model, NoneType):
            loss_log = LossLog()
            loss_log(model, self.LossDict, stage)

        # -- (7) return loss values for backprop: ----------------------
        return self.__total__()