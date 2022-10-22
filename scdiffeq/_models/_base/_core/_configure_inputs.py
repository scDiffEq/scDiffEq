

import inspect
import torch
from neural_diffeqs import NeuralSDE

from ._base_utility_functions import extract_func_kwargs
from ._prepare_lightning_data_module import prepare_LightningDataModule


class InputConfiguration:
    def __init__(self, func=None, adata=None, DataModule=None):
        self.__parse__(locals())

    def __parse__(self, kwargs, return_dict=False, ignore=["self"]):

        parsed_kwargs = {}
        non_null = []
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)
                parsed_kwargs[k] = v
                if not v is None:
                    non_null.append(k)

        setattr(self, "_attrs", parsed_kwargs.keys())
        setattr(self, "_non_null", non_null)
        if return_dict:
            return parsed_kwargs

    def _adata_only(self, use_key, time_key, w_key):

        self.DataModule = prepare_LightningDataModule(
            self.adata, use_key=use_key, time_key=time_key, w_key=w_key
        )
        self.func = NeuralSDE(state_size=self.DataModule.n_dims)

    def configure(self, use_key="X_pca", time_key="Time point", w_key="w", **kwargs):
        if self._non_null == ["adata"]:
            self._adata_only(use_key, time_key, w_key)

        elif self._non_null == ["DataModule"]:
            SDE_kwargs = extract_func_kwargs(NeuralSDE, kwargs)
            self.func = NeuralSDE(state_size=self.DataModule.n_dims, **SDE_kwargs)

        elif self._non_null == ["func"]:
            print(" - [ NOTE ] | PASS ADATA OR NEURAL DIFFEQ")

        elif self._non_null == ["func", "adata"]:
            self.DataModule = prepare_LightningDataModule(
                self.adata, use_key=use_key, time_key=time_key, w_key=w_key
            )

        elif self._non_null == ["adata", "DataModule"]:
            self.func = NeuralSDE(state_size=self.DataModule.n_dims, **SDE_kwargs)

        elif self._non_null == ["func", "DataModule"]:
            pass
        elif self._non_null == ["func", "adata", "DataModule"]:
            pass

    def pass_to_model(self, model):

        for attr in self._attrs:
            setattr(model, attr, getattr(self, attr))