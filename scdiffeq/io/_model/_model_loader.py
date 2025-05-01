# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import lightning
import logging
import pathlib
import pandas as pd
import torch
import torch_nets
import yaml

# -- import local dependencies: -----------------------------------------------
from ...core import lightning_models, utils, scDiffEq
from ._hparams import HParams
from ._project import Project
from ._checkpoint import Checkpoint
from ._version import Version

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- cls: ---------------------------------------------------------------------
class ModelLoader(ABCParse.ABCParse):
    def __init__(
        self,
        project_path=None,
        version=None,
        name_delim: str = ".",
        *args,
        **kwargs,
    ) -> None:
        """ """
        self.__parse__(locals())
        self._validate_version()

    @property
    def project(self) -> Project:
        if not hasattr(self, "_project"):
            self._project = Project(path=self._project_path)
        return self._project

    @property
    def _VERSION_KEY(self) -> str:

        if not hasattr(self, "_VERSION_KEY_PRIVATE"):
            if self._version is None:
                version_key = max(self.project._VERSION_PATHS)
                logger.info(f"Version not provided. Defaulting to: '{version_key}'")
                self._VERSION_KEY_PRIVATE = version_key
            else:
                self._VERSION_KEY_PRIVATE = f"version_{self._version}"

        return self._VERSION_KEY_PRIVATE

    @property
    def _VERSION_PATH(self) -> pathlib.Path:
        return self.project._VERSION_PATHS[self._VERSION_KEY]

    @property
    def version(self) -> Version:
        if not hasattr(self, "_VERSION"):
            self._VERSION = Version(path=self._VERSION_PATH)
        return self._VERSION

    @property
    def hparams(self) -> Dict[str, Any]:
        return self.version.hparams()

    @property
    def _MODEL_TYPE(self):
        name = self.hparams["name"]
        if ":" in name:  # for backwards-compatibility
            return name.split(":")[0].replace("-", "_")
        return name.split(self._name_delim)[0].replace("-", "_")

    @property
    def LightningModule(self):
        return getattr(lightning_models, self._MODEL_TYPE)

    @property
    def ckpt(self):
        return self.version.ckpts[self.epoch]

    @property
    def epoch(self) -> str:
        if not hasattr(self, "_epoch"):
            self._epoch = "last"
            logger.info(f"Epoch not provided. Defaulting to: '{self._epoch}'")
        return self._epoch

    def _validate_epoch(self) -> None:

        msg = f"ckpt at epoch: {self.epoch} does not exist. Choose from: {self.version._SORTED_CKPT_KEYS}"
        assert self.epoch in self.version.ckpts, msg

    def _validate_version(self) -> None:

        version_key = self._VERSION_KEY
        available_versions = [
            attr for attr in self.project.__dir__() if attr.startswith("version_")
        ]

        msg = f"{version_key} not found in project. Available versions: {available_versions}"
        assert version_key in self.project._VERSION_PATHS, msg

    def from_ckpt(self, ckpt_path: str, load_kwargs={"loading_existing": True}):
        return self.LightningModule.load_from_checkpoint(ckpt_path, **load_kwargs)

    def __call__(
        self,
        epoch: Optional[Union[str, int]] = None,
        plot_state_change: Optional[bool] = False,
        load_kwargs={"loading_existing": True},
        *args,
        **kwargs,
    ) -> lightning.LightningModule:
        """
        Parameters
        ----------
        epoch: Optional[Union[str, int]], default = None

        plot_state_change: Optional[bool], default = False

        Returns
        -------
        model:
        """

        self.__update__(locals())

        self._validate_epoch()
        ckpt_path = self.ckpt.path
        return self.LightningModule.load_from_checkpoint(ckpt_path, **load_kwargs)

def _inputs_from_ckpt_path(ckpt_path):
    """If you give the whole ckpt_path, you can derive the other inputs."""

    if isinstance(ckpt_path, str):
        ckpt_path = pathlib.Path(ckpt_path)

    project_path = ckpt_path.parent.parent.parent
    version = int(ckpt_path.parent.parent.name.split("_")[1])
    #     epoch = int(ckpt_path.name.split("-")[0].split("=")[1])

    return {"project_path": project_path, "version": version}  # , "epoch": epoch}


def load_diffeq(
    ckpt_path: Optional[Union[pathlib.Path, str]] = None,
    project_path: Optional[Union[pathlib.Path, str]] = None,
    version: Optional[int] = None,
    epoch: Optional[Union[int, str]] = None,
) -> lightning.LightningModule:
    """
    Load DiffEq from project_path, version [optional], and epoch [optional].

    Parameters
    ----------
    project_path : Union[pathlib.Path, str]
        Path to the project directory.
    version : Optional[int], default=None
        Version number of the model.
    epoch : Optional[Union[int, str]], default=None
        Epoch number of the model.

    Returns
    -------
    lightning.LightningModule
        Lightning differential equation model.
    """

    if not ckpt_path is None:
        inputs = _inputs_from_ckpt_path(ckpt_path)
        project_path = inputs["project_path"]
        version = inputs["version"]
    #         epoch = inputs['epoch']

    model_loader = ModelLoader(project_path=project_path, version=version)

    if not ckpt_path is None:
        return model_loader.from_ckpt(ckpt_path)
    return model_loader(epoch=epoch)


def load_model(
    adata: anndata.AnnData,
    ckpt_path: Optional[Union[pathlib.Path, str]] = None,
    project_path: Optional[Union[pathlib.Path, str]] = None,
    version: Optional[int] = None,
    epoch: Optional[Union[int, str]] = None,
    configure_trainer: Optional[bool] = False,
    #     silent: bool = False,
) -> scDiffEq:
    """
    Load scDiffEq model.

    Parameters
    ----------
    adata : anndata.AnnData
        adata object.
    ckpt_path : Optional[Union[pathlib.Path, str]], optional
        Checkpoint or ckpt_path are accepted, by default None
    project_path : Optional[Union[pathlib.Path, str]], optional
        Project path, by default None
    version : Optional[int], optional
        Version number, by default None
    epoch : Optional[Union[int, str]], optional
        Epoch number, by default None
    configure_trainer : Optional[bool], optional
        Indicate if trainer should be configured, by default False

    Returns
    -------
    scdiffeq.scDiffEq
        Loaded scDiffEq model.
    """

    if isinstance(ckpt_path, Checkpoint):
        ckpt_path = ckpt_path.path

    diffeq = load_diffeq(
        ckpt_path=ckpt_path,
        project_path=project_path,
        version=version,
        epoch=epoch,
    )

    # -- note: if ckpt_path is passed, get: project_path|version|epoch
    if not ckpt_path is None:
        project_path = ckpt_path.parent.parent.parent
        version = int(ckpt_path.parent.parent.name.split("version_")[-1])
        if ckpt_path.name == "last.ckpt":
            epoch = "last"
        else:
            epoch = int(ckpt_path.name.split("epoch=")[-1].split("-")[0])

    # ----------------------------------------------------------------

    sdq_info = {
        "ckpt": ckpt_path.name.split(".ckpt")[0],
        "version": version,
        "project": project_path.name,
    }

    # ----------------------------------------------------------------
    model = scDiffEq(**dict(diffeq.hparams))
    model._load_version = version
    model.configure_data(adata)
    model.configure_kNN()
    model.configure_model(
        diffeq,
        configure_trainer=configure_trainer,
        loading_existing=True,
    )

    PREVIOUS_METRICS_PATH = pathlib.Path(project_path).joinpath(
        f"version_{version}/metrics.csv"
    )
    PREVIOUS_METRICS = pd.read_csv(PREVIOUS_METRICS_PATH)

    model.DiffEq.COMPLETED_EPOCHS = int(PREVIOUS_METRICS["total_epochs"].max())

    model.adata.uns["sdq_info"] = sdq_info

    return model
