
import anndata
import ABCParse
from ._fate_perturbation_experiment import FatePerturbationExperiment

from typing import Optional, List

import tqdm.notebook


class FatePerturbationScreen(ABCParse.ABCParse):
    def __init__(self, *args, **kwargs):
        self.__parse__(locals())

        self.Results = {}

    @property
    def genes(self):
        if not hasattr(self, "_genes"):
            self._genes = self._adata.var_names.tolist()
        return self._genes

    def forward(self, gene):

        prtb_expt = FatePerturbationExperiment()
        result = prtb_expt(
            adata=self._adata,
            model=self._model,
            genes=ABCParse.as_list(gene),
            subset_key=self._subset_key,
            subset_val=self._subset_val,
            target_value=self._target_value,
            PCA=self._PCA,
        )
        self.Results.update({gene: result})

    def __call__(
        self,
        adata,
        model,
        target_value: float = 10,
        genes: Optional[List] = None,
        subset_key="Time point",
        subset_val=2,
        PCA: Optional = None,
        *args,
        **kwargs
    ):
        self.__update__(locals())

        for gene in tqdm.notebook.tqdm(self.genes):
            self.forward(gene)

        return self.Results
