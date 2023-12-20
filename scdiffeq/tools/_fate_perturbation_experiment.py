
import anndata
import ABCParse
import pandas as pd
import cell_perturb
import scipy.stats
import numpy as np

from ._simulation import simulate

from ._annotate_cell_state import annotate_cell_state
from ._annotate_cell_fate import annotate_cell_fate

from typing import Optional, List, Dict, Union


class PerturbationExperimentResult(ABCParse.ABCParse):
    def __init__(
        self,
        ctrl_result: anndata.AnnData,
        prtb_result: anndata.AnnData,
        *args,
        **kwargs,
    ):
        """
        Args:
            ctrl_result (anndata.AnnData)

            prtb_result (anndata.AnnData)
        """

        self.__parse__(locals())

    def _count_fates(self, result: pd.DataFrame):

        fate_counts = (
            result.obs.groupby("replicate")["fate"].value_counts().unstack().fillna(0)
        )
        return fate_counts.div(fate_counts.sum(1), axis=0).T

    def _forward(self, result: anndata.AnnData, key: str):
        return self._count_fates(result)

    #         result.columns = [f"{key}.{i}" for i in result.columns.tolist()]

    @property
    def ctrl(self) -> pd.DataFrame:
        if not hasattr(self, "_ctrl"):
            self._ctrl = self._forward(self._ctrl_result, key="ctrl")
        return self._ctrl

    @property
    def prtb(self) -> pd.DataFrame:
        if not hasattr(self, "_prtb"):
            self._prtb = self._forward(self._prtb_result, key="prtb")
        return self._prtb

    @property
    def _fates(self):
        return list(set(self.ctrl.index.tolist()).union(self.prtb.index.tolist()))

    def _zerofill(self, result_df: pd.DataFrame):
        result_t = result_df.copy().T
        for fate in self._fates:
            if not fate in result_t:
                result_t[fate] = 0

        return result_t

    def _compute_lfc(self, ctrl_t, prtb_t, constant: float = 1e-9):
        return (prtb_t + constant).div((ctrl_t + constant)).apply(np.log2)

    def _compute_pvals(self, ctrl_t, prtb_t):
        return pd.Series(
            {
                fate: scipy.stats.ttest_ind(
                    ctrl_t[fate], prtb_t[fate], equal_var=False
                )[1]
                for fate in self._fates
            }
        )

    def _compute_summary_statistics(self):
        
        ctrl_t = self._zerofill(self.ctrl)
        prtb_t = self._zerofill(self.prtb)
        self._lfc = self._compute_lfc(ctrl_t, prtb_t)
        pvals = self._compute_pvals(ctrl_t, prtb_t)
        
        lfc_pvals = pd.DataFrame([self._lfc.mean(), self._lfc.std(), pvals]).T
        lfc_pvals.columns = ["lfc", 'lfc_std', "pval"]

        return lfc_pvals
    
    @property
    def stats(self):
        if not hasattr(self, "_stats"):
            self._stats = self._compute_summary_statistics()
        return self._stats

    def __repr__(self):
        return "PerturbationExperimentResult"


class FatePerturbationExperiment(ABCParse.ABCParse):
    def __init__(
        self,
        seed: int = 0,
        use_key: str = "X_scaled",
        replicates: int = 5,
        N: int = 200,
        time_key: str = "t",
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

    @property
    def _PERTURBATION_INIT_KWARGS(self):
        return ABCParse.function_kwargs(
            func=cell_perturb.Perturbation.__init__, kwargs=self._PARAMS
        )

    @property
    def _PERTURBATION_CALL_KWARGS(self):
        return ABCParse.function_kwargs(
            func=cell_perturb.Perturbation.__call__, kwargs=self._PARAMS
        )

    @property
    def adata_prtb(self):
        if not hasattr(self, "_adata_prtb"):
            self._perturbation = cell_perturb.Perturbation(
                **self._PERTURBATION_INIT_KWARGS
            )
            self._adata_prtb = self._perturbation(**self._PERTURBATION_CALL_KWARGS)
        return self._adata_prtb

    def _subset_final_state(self, adata_sim):
        t = adata_sim.obs[self._time_key]
        return adata_sim[t == t.max()].copy()

    def forward(self):
        adata_sim_prtb = simulate(
            adata=self.adata_prtb,
            model=self._model,
            use_key="X_pca_prtb",
        )
        adata_sim_ctrl = simulate(
            adata=self.adata_prtb,
            model=self._model,
            use_key="X_pca_ctrl",
        )
        prtb = self._subset_final_state(adata_sim_prtb)
        ctrl = self._subset_final_state(adata_sim_ctrl)
        
        rep = self.adata_prtb.obs["replicate"].values
        
        prtb.obs["replicate"] = rep
        ctrl.obs["replicate"] = rep

        annotate_cell_state(prtb, self._model.kNN, silent=True)
        annotate_cell_fate(prtb, silent=True)

        annotate_cell_state(ctrl, self._model.kNN, silent=True)
        annotate_cell_fate(ctrl, silent=True)
        
        return ctrl, prtb

    def __call__(
        self,
        adata: anndata.AnnData,
        model: "scdiffeq.scDiffEq",
        genes: List[str],
        subset_key: str,
        subset_val: str,
        target_value: float = 10,
        PCA: Optional = None,
        *args,
        **kwargs,
    ):
        self.__update__(locals())

        self.ctrl_result, self.prtb_result = self.forward()
        
        return PerturbationExperimentResult(self.ctrl_result, self.prtb_result)
