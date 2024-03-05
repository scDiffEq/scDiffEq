
import anndata
import ABCParse
import torch


from ._fate_perturbation_experiment import FatePerturbationExperiment

from typing import Optional, List

import tqdm as tqdm_sh
import tqdm.notebook


class FatePerturbationScreen(ABCParse.ABCParse):
    def __init__(
        self,
        seed: int = 0,
        use_key: str = "X_scaled",
        replicates: int = 5,
        N: int = 200,
        time_key: str = "t",
        nb: bool = True,
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

        self.Results = {}

    @property
    def genes(self):
        if not hasattr(self, "_genes"):
            self._genes = self._adata.var_names.tolist()
        return self._genes

    def forward(self, gene):

        prtb_expt = FatePerturbationExperiment(
            seed = self._seed,
            use_key = self._use_key,
            replicates = self._replicates,
            N = self._N,
            time_key = self._time_key,
        )
        result = prtb_expt(
            adata=self._adata,
            model=self._model,
            genes=ABCParse.as_list(gene),
            t_sim=self._t_sim,
            subset_key=self._subset_key,
            subset_val=self._subset_val,
            target_value=self._target_value,
            obs_key=self._obs_key,
            PCA=self._PCA,
        )
        self.Results.update({gene: result})

    @property
    def _progress_bar(self):
        if self._nb:
            return tqdm.notebook.tqdm(self.genes)
        return tqdm_sh.tqdm(self.genes)
    
    def __call__(
        self,
        adata: anndata.AnnData,
        model,
        obs_key: str,
        t_sim: torch.Tensor,
        target_value: float = 10,
        genes: Optional[List] = None,
        subset_key="Time point",
        subset_val=2,
        PCA: Optional = None,
        *args,
        **kwargs
    ):
        self.__update__(locals())

        for gene in self._progress_bar: # tqdm.notebook.tqdm(self.genes):
            self.forward(gene)

        return self.Results
