# -- import packages: ---------------------------------------------------
import anndata
import logging
import numpy as np
import sklearn.decomposition
import torch

# -- initialize logger: ----------------------------------------------------
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)

# -- import local dependencies: ------------------------------------------
from ._fate_perturbation_experiment import PerturbationExperimentResult
from ._perturb import perturb

# -- set typing: -----------------------------------------------------------
from typing import Dict, List, Optional


# -- API-facing function: -------------------------------------------------
def perturb_scan_z_range(
    adata: anndata.AnnData,
    model: "scdiffeq.scDiffEq",
    obs_key: str,
    subset_key: str,
    subset_val: str,
    genes: List[str],
    t_sim: torch.Tensor,
    z_range: np.ndarray,
    PCA: Optional[sklearn.decomposition.PCA] = None,
    seed: int = 0,
    gene_id_key: str = "gene_ids",
    use_key: str = "X_scaled",
    replicates: int = 5,
    N: int = 2000,
    save_simulation: bool = False,
    *args,
    **kwargs,
) -> Dict[float, PerturbationExperimentResult]:
    """Perturb a population of cells and simulate the resulting trajectories over a range of target z-score values.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to perturb.
    model : scdiffeq.scDiffEq
        The scDiffEq model to perturb.
    obs_key : str
        Key in `adata.obs` to use for observations.
    subset_key : str
        Key in `adata.obs` to subset the data.
    subset_val : str
        Value in `subset_key` to subset the data.
    genes : List[str]
        List of genes to perturb.
    t_sim : torch.Tensor
        Simulation time points.
    z_range : np.ndarray
        Array of target z-score values for the perturbation.
    PCA : optional
        PCA object for dimensionality reduction. Default is None.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    gene_id_key : str, optional
        Key in `adata.var` for gene identifiers. Default is "gene_ids".
    use_key : str, optional
        Key in `adata.obsm` for the data to use. Default is "X_scaled".
    replicates : int, optional
        Number of replicates for the perturbation. Default is 5.
    N : int, optional
        Number of cells to perturb. Default is 2000.
    save_simulation : bool, optional
        Whether to save the simulation results. Default is False.
    *args :
        Additional arguments.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    Dict[float, PerturbationExperimentResult]
        A dictionary where keys are z-score values and values are the results of the perturbation experiment.
    """

    Perturbed = {}
    for z in z_range:
        Perturbed[z] = perturb(
            adata=adata,
            model=model,
            target_value=z,
            seed=seed,
            N=N,
            t_sim=t_sim,
            obs_key=obs_key,
            subset_key=subset_key,
            subset_val=subset_val,
            gene_id_key=gene_id_key,
            genes=genes,
            replicates=replicates,
            PCA=PCA,
            *args,
            **kwargs,
        )
        logger.info(f"Perturbed {len(genes)} genes (Z={z})")
    return Perturbed
