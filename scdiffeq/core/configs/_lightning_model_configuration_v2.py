import ABCParse

from .. import lightning_models, utils

# -- import packages: ----------------------------------------------------------
class LightningModelConfiguration(ABCParse.ABCParse):
    _potential_types = {
        "fixed": "FixedPotential",
        "prior": "PriorPotential",
    }

    def __init__(
        self,
        data_dim,
        latent_dim: int = 20,
        DiffEq_type: str = "SDE",
        potential_type="prior",
    ):

        self.__parse__(locals(), public=[None])

    @property
    def available_lightning_models(self):
        return [
            attr
            for attr in lightning_models.__dir__()
            if attr.startswith("Lightning")
        ]

    @property
    def DiffEq_type(self):
        return "".join(["Lightning", self._DiffEq_type])

    @property
    def use_vae(self):
        return self._data_dim > self._latent_dim

    @property
    def potential_type(self):
        if self._potential_type:
            return self._potential_types[self._potential_type]

    def __call__(self, kwargs):

        _model = [self.DiffEq_type]

        if self.use_vae:
            _model.append("VAE")

        if self.potential_type:
            _model.append(self.potential_type)

        _model = "_".join(_model)

        if _model in self.available_lightning_models:
            lit_model = getattr(lightning_models, _model)
            model_kwargs = utils.function_kwargs(func=lit_model.__init__, kwargs=kwargs)
            # data_dim=self._data_dim, latent_dim=self._latent_dim, 
            return lit_model(data_dim = self._data_dim, **model_kwargs)
