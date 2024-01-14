

import ABCParse
import anndata
import pathlib
import lightning
import torch
import torch_nets
import yaml
import pandas as pd

from ...core import lightning_models, utils, scDiffEq


from typing import Optional, Union

from ._hparams import HParams
from ._project import Project
from ._checkpoint import Checkpoint
from ._version import Version


class ModelLoader(ABCParse.ABCParse):
    def __init__(self, project_path = None, version = None, name_delim: str = ".", *args, **kwargs):
        """ """
        self.__parse__(locals())
        self._INFO = utils.InfoMessage()
        self._validate_version()

    @property
    def project(self):
        if not hasattr(self, "_project"):
            self._project = Project(path=self._project_path)
        return self._project

    @property
    def _VERSION_KEY(self):

        if not hasattr(self, "_VERSION_KEY_PRIVATE"):
            if self._version is None:
                version_key = max(self.project._VERSION_PATHS)
                self._INFO(f"Version not provided. Defaulting to: '{version_key}'")
                self._VERSION_KEY_PRIVATE = version_key
            else:
                self._VERSION_KEY_PRIVATE = f"version_{self._version}"

        return self._VERSION_KEY_PRIVATE

    @property
    def _VERSION_PATH(self):
        return self.project._VERSION_PATHS[self._VERSION_KEY]

    @property
    def version(self):
        if not hasattr(self, "_VERSION"):
            self._VERSION = Version(path=self._VERSION_PATH)
        return self._VERSION

    @property
    def hparams(self):
        return self.version.hparams()

    @property
    def _MODEL_TYPE(self):
        name = self.hparams["name"]
        if ":" in name:
            return name.split(":")[0].replace("-", "_")
        return name.split(self._name_delim)[0].replace("-", "_")

    @property
    def LightningModule(self):
        return getattr(lightning_models, self._MODEL_TYPE)
#         if not hasattr(self, "_model"):
#             LitModel = getattr(lightning_models, self._MODEL_TYPE)
#             self._model = LitModel(**self.hparams)
#         return self._model

    @property
    def ckpt(self):
        return self.version.ckpts[self.epoch]
    
    @property
    def epoch(self):
        if not hasattr(self, "_epoch"):
            self._epoch = "last"
            self._INFO(f"Epoch not provided. Defaulting to: '{self._epoch}'")
        return self._epoch

    def _validate_epoch(self):
            
        msg = f"ckpt at epoch: {self.epoch} does not exist. Choose from: {self.version._SORTED_CKPT_KEYS}"
        assert self.epoch in self.version.ckpts, msg

    def _validate_version(self):
        
        version_key = self._VERSION_KEY
        available_versions = [
            attr for attr in self.project.__dir__() if attr.startswith("version_")
        ]

        msg = f"{version_key} not found in project. Available versions: {available_versions}"
        assert version_key in self.project._VERSION_PATHS, msg

    def from_ckpt(self, ckpt_path: str):
        return self.LightningModule.load_from_checkpoint(ckpt_path)
    
    def __call__(
        self,
        epoch: Optional[Union[str, int]] = None,
        plot_state_change: Optional[bool] = False,
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
        model = self.LightningModule.load_from_checkpoint(ckpt_path)
        # self.model

#         if plot_state_change:
#             torch_nets.pl.weights_and_biases(model.state_dict())
            
#         ckpt_path = self.ckpt.path
#         self._INFO(f"Loading model from ckpt: \n\t'{ckpt_path}'")
#         model = model.load_from_checkpoint(ckpt_path)
        
#         if plot_state_change:
#             torch_nets.pl.weights_and_biases(model.state_dict())

        return model

def _inputs_from_ckpt_path(ckpt_path):
    """If you give the whole ckpt_path, you can derive the other inputs."""
    
    if isinstance(ckpt_path, str):
        ckpt_path = pathlib.Path(ckpt_path)

    project_path = ckpt_path.parent.parent.parent
    version = int(ckpt_path.parent.parent.name.split("_")[1])
#     epoch = int(ckpt_path.name.split("-")[0].split("=")[1])
    
    return {"project_path": project_path, "version": version} # , "epoch": epoch}
    
def load_diffeq(
    ckpt_path: Optional[Union[pathlib.Path, str]] = None,
    project_path: Optional[Union[pathlib.Path, str]] = None,
    version: Optional[int] = None,
    epoch: Optional[Union[int, str]] = None,
) -> lightning.LightningModule:

    """
    Load DiffEq from project_path, version [optional], and epoch [optional].
    
    Args:
        project_path (Union[pathlib.Path, str])

        version (Optional[int]): **Default** = None

        epoch (Optional[Union[int, str]]): **Default** = None

    
    Returns:
        DiffEq (lightning.LightningModule): lightning differential equation model.
    """
    
    if not ckpt_path is None:
        inputs = _inputs_from_ckpt_path(ckpt_path)
        project_path = inputs['project_path']
        version = inputs['version']
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
):
    
    """Load scDiffEq model.
    
    Args:
        adata (anndata.AnnData): adata object.
        
        ckpt_path (Optional[Union[pathlib.Path, str]]): description. **Default** = None
        
        project_path (Optional[Union[pathlib.Path, str]]): description. **Default** = None
        
        version (Optional[int]): description. **Default** = None
        
        epoch (Optional[Union[int, str]]): description. **Default** = None
        
        configure_trainer (Optional[bool]): indicate if trainer should be configured. **Default** = False.
    
    Returns:
        scdiffeq.scDiffEq
    """

    diffeq = load_diffeq(
        ckpt_path=ckpt_path,
        project_path=project_path,
        version=version,
        epoch=epoch,
    )
    model = scDiffEq(**dict(diffeq.hparams))
    model.configure_data(adata)
    model.configure_kNN()
    model.configure_model(diffeq, configure_trainer = configure_trainer)

    return model
