

import ABCParse
import anndata
import pathlib
import lightning
import torch
import torch_nets
import yaml


from ..core import lightning_models, utils, scDiffEq


from typing import Optional, Union


class Project(ABCParse.ABCParse):
    def __init__(self, path=pathlib.Path("./").absolute(), *args, **kwargs):

        self.__parse__(locals())

        for k, v in self._VERSION_PATHS.items():
            setattr(self, k, v)

    @property
    def _PROJECT_PATH(self):
        if isinstance(self._path, pathlib.Path):
            return self._path
        elif isinstance(self._path, str):
            return pathlib.Path(self._path)
        else:
            raise TypeError("arg: `path` must be of type: [pathlib.Path, str]")

    @property
    def _VERSION_PATHS(self):
        version_paths = sorted(list(self._PROJECT_PATH.glob("version_*")))
        return {path.name: path for path in version_paths}


class HyperParams(ABCParse.ABCParse):
    def __init__(self, yaml_path):

        self.__configure__(locals())

    def _read(self):
        if not hasattr(self, "_file"):
            self._yaml_file = yaml.load(open(self._yaml_path), Loader=yaml.Loader)

    def __configure__(self, kwargs, private=["yaml_path"]):

        self.__parse__(kwargs, private=private)
        self._read()
        for key, val in self._yaml_file.items():
            setattr(self, key, val)

    @property
    def _ATTRS(self):
        self._attrs = {
            attr: getattr(self, attr)
            for attr in self.__dir__()
            if not attr[0] in ["_", "a"]
        }
        return self._attrs

    def __repr__(self):

        """Return a readable representation of the discovered hyperparameters"""

        string = "HyperParameters\n"
        for attr, val in self._ATTRS.items():
            string += "\n  {:<34}: {}".format(attr, val)

        return string

    def __call__(self):
        return self._ATTRS


class Checkpoint(ABCParse.ABCParse):
    def __init__(self, path: pathlib.Path, *args, **kwargs):
        self.__parse__(locals(), public=["path"])

    @property
    def _fname(self):
        return self.path.name.split(".")[0]

    @property
    def epoch(self):
        if self._fname != "last":
            return int(self._fname.split("=")[1].split("-")[0])
        return self._fname

    @property
    def state_dict(self):
        if not hasattr(self, "_ckpt"):
            self._state_dict = torch.load(self.path)  # ["state_dict"]
        return self._state_dict

    def __repr__(self):
        return f"ckpt epoch: {self.epoch}"


class Version(ABCParse.ABCParse):
    def __init__(self, path = None, *args, **kwargs):

        self.__parse__(locals())

    @property
    def _CONTENTS(self):
        return list(version_path.glob("*"))

    @property
    def hparams(self):
        hparams_path = self._path.joinpath("hparams.yaml")
        if hparams_path.exists():
            return HyperParams(hparams_path)

    @property
    def metrics_df(self):
        metrics_path = self._path.joinpath("metrics.csv")
        if metrics_path.exists():
            return pd.read_csv(metrics_path)

    @property
    def _CKPT_PATHS(self):
        _ckpt_paths = list(self._path.joinpath("checkpoints").glob("*"))
        return [pathlib.Path(path) for path in _ckpt_paths]

    @property
    def _SORTED_CKPT_KEYS(self):
        epochs = list(self.ckpts.keys())
        _epochs = sorted([epoch for epoch in epochs if epoch != "last"])
        if "last" in epochs:
            _epochs.append("last")
        return _epochs

    @property
    def ckpts(self):
        if not hasattr(self, "_CHECKPOINTS"):
            self._CHECKPOINTS = {}
            for ckpt_path in self._CKPT_PATHS:
                ckpt = Checkpoint(ckpt_path)
                self._CHECKPOINTS[ckpt.epoch] = ckpt
        return self._CHECKPOINTS


class ModelLoader(ABCParse.ABCParse):
    def __init__(self, project_path = None, version = None, *args, **kwargs):
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
        return self.hparams["name"].split(":")[0].replace("-", "_")

    @property
    def model(self):
        if not hasattr(self, "_model"):
            LitModel = getattr(lightning_models, self._MODEL_TYPE)
            self._model = LitModel(**self.hparams)
        return self._model

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

        model = self.model

        if plot_state_change:
            torch_nets.pl.weights_and_biases(model.state_dict())
            
        ckpt_path = self.ckpt.path
        self._INFO(f"Loading model from ckpt: \n\t'{ckpt_path}'")
        model = model.load_from_checkpoint(ckpt_path)
        
        if plot_state_change:
            torch_nets.pl.weights_and_biases(model.state_dict())

        return model
    
def load_diffeq(
    project_path: Union[pathlib.Path, str],
    version: Optional[int] = None,
    epoch: Optional[Union[int, str]] = None,
) -> lightning.LightningModule:

    """
    Load DiffEq from project_path, version [optional], and epoch [optional].
    
    Parameters
    ----------
    project_path: Union[pathlib.Path, str]

    version: Optional[int], default = None

    epoch: Optional[Union[int, str]], default = None

    Returns
    -------
    DiffEq: lightning.LightningModule
    """

    model_loader = ModelLoader(project_path=project_path, version=version)
    return model_loader(epoch=epoch)

def load_model(
    adata: anndata.AnnData,
    project_path: Union[pathlib.Path, str],
    version: Optional[int] = None,
    epoch: Optional[Union[int, str]] = None,
):

    diffeq = load_diffeq(
        project_path=project_path,
        version=version,
        epoch=epoch,
    )
    model = scDiffEq(**dict(diffeq.hparams))
    model.configure_data(adata)
    model.configure_kNN()
    model.configure_model(diffeq)

    return model
