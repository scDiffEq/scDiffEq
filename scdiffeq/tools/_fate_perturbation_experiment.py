
# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import cell_perturb
import lightning
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import scipy.stats
import torch


# -- initialize logger: -------------------------------------------------------
logger = logging.getLogger(name=__name__)


# -- import local dependencies: -----------------------------------------------
from ._annotate_cell_state import annotate_cell_state
from ._annotate_cell_fate import annotate_cell_fate
from ._simulation import simulate


# -- set typing: --------------------------------------------------------------
from typing import Dict, List, Optional, Tuple, Union


# -- Operational class: -------------------------------------------------------
class PerturbationExperimentResult(ABCParse.ABCParse):
    """Container for the results of a perturbation of experiment. Both the control
    and perturbed arms of the experiment are given as input.
    
    Generally the user interacts with this class, but does not instantiate it. Instead,
    it is instantiated through the output of ``FatePerturbationExperiment``.

    Attributes:
        ctrl
        
        prtb
        
        stats
    """
    def __init__(
        self,
        ctrl_result: anndata.AnnData,
        prtb_result: anndata.AnnData,
        *args,
        **kwargs,
    ) -> None:
        """ Initialize the ``PerturbationExperimentResult`` class.
        
        Args:
            ctrl_result (anndata.AnnData) The resulting AnnData object containing the final state of the simulated control, over each replicate.

            prtb_result (anndata.AnnData). The resulting AnnData object containing the final state of the simulated perturbation, over each replicate.
            
        Returns:
            None
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


# -- API-facing operational class: --------------------------------------------
class FatePerturbationExperiment(ABCParse.ABCParse):
    """Container class for an expression perturbation experiment, designed to facilitate
    the analysis of gene expression perturbations and their effects on cell fate and state.
    
    Inherits from ABCParse for abstract base class parsing functionality.
    
    Attributes:
        
    """
    def __init__(
        self,
        seed: int = 0,
        use_key: str = "X_scaled",
        replicates: int = 5,
        N: int = 200,
        time_key: str = "t",
        save_simulation: bool = False,
        save_path: Optional[pathlib.Path] = pathlib.Path("./scdiffeq_simulations"),
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the FatePerturbationExperiment object.
        
        Args:
            seed (int): Seed for random number generation to ensure reproducibility.

            use_key (str): Key to use for the expression data within the AnnData object.

            replicates (int): Number of replicates to consider in the perturbation experiment.

            N (int): The number of cells to simulate.

            time_key (str): Key to access time-related data within the AnnData object.

            *args, **kwargs: Additional arguments and keyword arguments for flexibility and future extensions.
        """
        self.__parse__(locals())

    @property
    def _PERTURBATION_INIT_KWARGS(self) -> Dict:
        """Retrieves the function keyword arguments for initializing the Perturbation object.
        (i.e., ``cell_perturb.Perturbation.__init__``).
        
        Returns:
            Dict: A dictionary of keyword arguments used for Perturbation object initialization.
        """
        return ABCParse.function_kwargs(
            func=cell_perturb.Perturbation.__init__, kwargs=self._PARAMS
        )

    @property
    def _PERTURBATION_CALL_KWARGS(self) -> Dict:
        """Retrieves the function keyword arguments for calling the Perturbation object
        (i.e., ``cell_perturb.Perturbation.__call__``).
        
        Returns:
            Dict: A dictionary of keyword arguments used for calling the Perturbation object.
        """
        return ABCParse.function_kwargs(
            func=cell_perturb.Perturbation.__call__, kwargs=self._PARAMS
        )

    @property
    def adata_prtb(self) -> anndata.AnnData:
        """Lazily loads or generates the AnnData object resulting from perturbation.
        
        Returns:
            adata_prtb (anndata.AnnData): The AnnData object after applying perturbation.
        """
        if not hasattr(self, "_adata_prtb"):
            self._perturbation = cell_perturb.Perturbation(
                **self._PERTURBATION_INIT_KWARGS
            )
            self._adata_prtb = self._perturbation(**self._PERTURBATION_CALL_KWARGS)
        return self._adata_prtb

    def _subset_final_state(self, adata_sim) -> anndata.AnnData:
        """Extracts the subset of the AnnData object corresponding to the final state of simulation.
        
        Args:
            adata_sim (anndata.AnnData): The simulated AnnData object.
        
        Returns:
            adata_final (anndata.AnnData): A subset of the AnnData object at its final state.
        """

        t = adata_sim.obs[self._time_key]
        return adata_sim[t == t.max()].copy()
    
    @property
    def DiffEq(self) -> lightning.LightningModule:
        """Accessor for the differential equation model used in the simulation.

        Returns:
            DiffEq (lightning.LightningModule)
        """
        if isinstance(self._model.DiffEq, lightning.LightningModule):
            return self._model.DiffEq
        elif isinstance(self._model, lightning.LightningModule):
            return self._model    

    def forward(self) -> Tuple[anndata.AnnData, anndata.AnnData]:
        
        
        """
        Executes the perturbation experiment, comparing control and perturbed conditions.
        
        Simulates, subsets the final simulated state, annotates respective replicates, then annotates
        cell states/fates using the given kNN.
        
        Returns:
            [adata_ctrl, adata_prtb] (Tuple[anndata.AnnData, anndata.AnnData]): A tuple containing AnnData objects for control and perturbation experiments, respectively.
        """
        
        adata_sim_prtb = simulate(
            adata=self.adata_prtb,
            diffeq=self.DiffEq,
            use_key="X_pca_prtb",
            t = self._t_sim,
#             time_key = self._time_key,
        )
        adata_sim_ctrl = simulate(
            adata=self.adata_prtb,
            diffeq=self.DiffEq,
            use_key="X_pca_ctrl",
            t = self._t_sim,
#             time_key = self._time_key,
        )
        
        
        if self._save_simulation:
            
            if not self._save_path.exists():
                os.mkdir(self._save_path)
            
            self.adata_sim_prtb = adata_sim_prtb
            self.adata_sim_ctrl = adata_sim_ctrl
            _genes_ = "_".join(self._genes)
            
            del self.adata_sim_prtb.uns['sim_idx']
            del self.adata_sim_ctrl.uns['sim_idx']
            
            self.adata_sim_prtb.write_h5ad(self._save_path.joinpath(f"adata.{_genes_}.prtb.h5ad"))
            self.adata_sim_ctrl.write_h5ad(self._save_path.joinpath(f"adata.{_genes_}.ctrl.h5ad"))
            logger.info(f"Perturbed simulations saved to: {self._save_path}")
            
        
        prtb = self._subset_final_state(adata_sim_prtb)
        ctrl = self._subset_final_state(adata_sim_ctrl)
        
        rep = self.adata_prtb.obs["replicate"].values
        
        prtb.obs["replicate"] = rep
        ctrl.obs["replicate"] = rep
        
        annotate_cell_state(prtb, kNN = self._model.kNN, obs_key = self._obs_key, silent=True)
        annotate_cell_fate(prtb, state_key = self._obs_key, silent=True)

        annotate_cell_state(ctrl, kNN = self._model.kNN, obs_key = self._obs_key, silent=True)
        annotate_cell_fate(ctrl, state_key = self._obs_key, silent=True)
        
        return ctrl, prtb

    def __call__(
        self,
        adata: anndata.AnnData,
        model: "scdiffeq.scDiffEq",
        t_sim: torch.Tensor,
        obs_key: str,
        genes: List[str],
        subset_key: str,
        subset_val: str,
        target_value: float = 10,
        PCA: Optional = None,
        *args,
        **kwargs,
    ):
        """
        Run perturbation screen.
        
        Args:
            adata (anndata.AnnData): adata obj.
            
            
            model ("scdiffeq.scDiffEq"): scDiffEq model.
            
            genes (List[str]): Genes over which screen should be run.
        
            subset_key (str):
            
            subset_val (str):
        
            target_value (float): Z-score value at which perturbation should be set. **Default**: 10
            
            PCA (Optional[sklearn.decomposition.PCA]: PCA model for transforming expression to model input. **Default**: None.
        
        Returns:
            PerturbationExperimentResult
        """
        self.__update__(locals())

        self.ctrl_result, self.prtb_result = self.forward()
        
        return PerturbationExperimentResult(self.ctrl_result, self.prtb_result)
