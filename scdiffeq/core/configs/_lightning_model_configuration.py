import ABCParse
import os

from .. import lightning_models, utils


NoneType = type(None)
# figure out how to return model from file wihtout doing anyhhing else - fast, easy
#     def load_from_ckpt(self):
        
#         return 
#         self.DiffEq = 
#         def load_model(self):
#         diffeq = sdq.core.lightning_models.LightningSDE_FixedPotential.

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
        fate_bias_csv_path = None,
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
        
    @property
    def fate_bias_aware(self):
        if isinstance(self._fate_bias_csv_path, NoneType):
            return False
        else:
            if os.path.exists(self._fate_bias_csv_path):
                return True
            else:
                raise ValueError("Path to fate_bias.csv was passed though not found.")

    @property
    def _USE_CKPT(self):
        return isinstance(self._ckpt_path, str)
    
    def __call__(self, kwargs, ckpt_path = None):
        
        self._ckpt_path = ckpt_path

        _model = [self.DiffEq_type]

        if self.use_vae:
            _model.append("VAE")

        if self.potential_type:
            _model.append(self.potential_type)
            
        if self.fate_bias_aware:
            _model.append("FateBiasAware")

        _model = "_".join(_model)

        if _model in self.available_lightning_models:
            lit_model = getattr(lightning_models, _model)
            
            if self._USE_CKPT:
                return lit_model.load_from_checkpoint(self._ckpt_path)
            
            model_kwargs = utils.function_kwargs(func=lit_model.__init__, kwargs=kwargs)
            
            return lit_model(data_dim = self._data_dim, **model_kwargs)
        
        raise ValueError(f"Configuration tried: {_model} - this does not exist as an available model.")
