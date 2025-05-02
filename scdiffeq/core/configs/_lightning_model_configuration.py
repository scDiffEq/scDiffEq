# -- import packages: ---------------------------------------------------------
import ABCParse
import logging
import os

# -- import local dependencies: -----------------------------------------------
from .. import lightning_models, utils

# -- set typing: --------------------------------------------------------------
from typing import Dict, Optional

# -- setup logging: -----------------------------------------------------------
logger = logging.getLogger(__name__)


# -- operational class: -------------------------------------------------------
class LightningModelConfiguration(ABCParse.ABCParse):
    """ """

    _potential_types = {
        "fixed": "FixedPotential",
        "prior": "PriorPotential",
    }

    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        DiffEq_type: str = "SDE",
        potential_type: str = "fixed",
        fate_bias_csv_path=None,
        velocity_ratio_params: Dict = None,
    ):
        """ """

        self.__parse__(locals(), public=[None])

    @property
    def available_lightning_models(self):
        return [
            attr for attr in lightning_models.__dir__() if attr.startswith("Lightning")
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
    def fate_bias_aware(self) -> bool:
        """
        First checks if the past is passed. If so, checks if it exists.
        If passed but path does not exist, raises a ValueError.
        """
        if not self._fate_bias_csv_path is None:
            if not os.path.exists(self._fate_bias_csv_path):
                raise ValueError("Path to fate_bias.csv was passed though not found.")
            else:
                return True
        else:
            return False

    @property
    def _USE_CKPT(self) -> bool:
        return not self._ckpt_path is None

    def __call__(
        self,
        kwargs: Dict,
        ckpt_path: Optional[str] = None,
        loading_existing: bool = False,
    ):

        self._ckpt_path = ckpt_path

        _model = [self.DiffEq_type]

        if self.use_vae:
            _model.append("VAE")

        if self.potential_type:
            _model.append(self.potential_type)

        if self.fate_bias_aware:
            _model.append("FateBiasAware")

        if (self.DiffEq_type == "SDE") and self._velocity_ratio_params:
            _model.append("RegularizedVelocityRatio")

        _model = "_".join(_model)

        if _model in self.available_lightning_models:
            lit_model = getattr(lightning_models, _model)

            if self._USE_CKPT:
                return lit_model.load_from_checkpoint(self._ckpt_path)

            model_kwargs = utils.function_kwargs(func=lit_model.__init__, kwargs=kwargs)
            model_kwargs["loading_existing"] = loading_existing
            logger.debug(model_kwargs)
            return lit_model(**model_kwargs)  # data_dim = self._data_dim,

        raise ValueError(
            f"Configuration tried: {_model} - this does not exist as an available model."
        )
