# -- import packages: --------------------------------------------------------
import anndata
import lightning
import logging

# -- import local dependencies: -----------------------------------------------
from .. import configs, utils

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- set type hints: ----------------------------------------------------------
from typing import Dict, Optional

# -- configure logging: -------------------------------------------------------
logger = logging.getLogger(__name__)


# -- mix-in cls: --------------------------------------------------------------
class ModelConfigMixIn(object):
    """configure_model and configure_data can/should be accessed from the public handle"""

    def _configure_trainer_generator(self) -> None:
        """ """

        self.TrainerGenerator = configs.LightningTrainerConfiguration(
            save_dir=self._name,
        )
        logger.debug(f"TrainerGenerator configured: {self._name}")

    def configure_model(
        self,
        DiffEq: Optional[lightning.LightningModule] = None,
        configure_trainer: bool = True,
        loading_existing: bool = False,
    ) -> None:
        if DiffEq is None:
            self._LitModelConfig = configs.LightningModelConfiguration(
                data_dim=self._data_dim,
                latent_dim=self._latent_dim,
                DiffEq_type=self._DiffEq_type,
                potential_type=self._potential_type,
                fate_bias_csv_path=self._fate_bias_csv_path,
                velocity_ratio_params=self._velocity_ratio_params,
            )

            if hasattr(self, "reducer"):
                self._PARAMS["PCA"] = self.reducer.PCA
            DiffEq = self._LitModelConfig(
                self._PARAMS, self._ckpt_path, loading_existing=loading_existing
            )

        self.DiffEq = DiffEq
        self._name = self.DiffEq.hparams.name

        logger.debug(f"Using the specified parameters, {self.DiffEq} has been called.")
        self._component_loader = utils.FlexibleComponentLoader(self)

        lightning.seed_everything(self._seed)

        # -- Step 5: configure bridge to lightning logger ----------------------
        # was its own step before: now in-line here, since it
        # doesn't make sense to separate it, functionally
        self._LOGGING = utils.LoggerBridge(self.DiffEq)

        if configure_trainer:
            self._configure_trainer_generator()

    def configure_data(self, adata: anndata.AnnData) -> None:
        """ """
        self.adata = adata.copy()

        self._DATA_CONFIG = configs.DataConfiguration()
        self._DATA_CONFIG(scDiffEq=self)
        logger.info(f"Input data configured.")

    def __config__(self, kwargs: Dict):
        """ """
        # -- Step 1: parse kwargs, set up info msg -----------------------------
        self.__parse__(kwargs, public=[None], ignore=["adata"])

        # -- Step 2: configure data ----------------------------------------------
        if not kwargs["adata"] is None:
            # if adata is given, triggers 2 through 4
            self.configure_data(adata=kwargs["adata"])

            # -- Step 3: configure kNN --------------------------------------------
            if self._PARAMS["build_kNN"]:
                self._PARAMS["kNN"] = self.kNN

            # -- Step 4: configure model -------------------------------------------
            self.configure_model(DiffEq=None, configure_trainer=True)

            # -- Step 5: extras (was step 6): ---------------------------------------


#             if kwargs["reduce_dimensions"]:
#                 self._configure_dimension_reduction()
