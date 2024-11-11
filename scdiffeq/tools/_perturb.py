# -- import packages: ---------------------------------------------------------
import anndata
from scdiffeq.tools._fate_perturbation_experiment import PerturbationExperimentResult
import torch


# -- import local dependencies: -----------------------------------------------
from ._fate_perturbation_experiment import FatePerturbationExperiment


# -- set typing: --------------------------------------------------------------
from typing import List, Optional


# -- API-facing function: -----------------------------------------------------
def perturb(
    adata: anndata.AnnData,
    model: "scdiffeq.scDiffEq",
    t_sim: torch.Tensor,
    obs_key: str,
    genes: List[str],
    subset_key: str,
    subset_val: str,
    gene_id_key: str = "gene_ids",
    target_value: float = 10,
    PCA: Optional = None,
    seed: int = 0,
    use_key: str = "X_scaled",
    replicates: int = 5,
    N: int = 200,
    save_simulation: bool = False,
    *args,
    **kwargs,
) -> PerturbationExperimentResult:
    """
    Perturb a population of cells and simulate the resulting trajectories.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to perturb.
    model : scdiffeq.scDiffEq
        The scDiffEq model to perturb.
    t_sim : torch.Tensor
        Simulation time points.
    obs_key : str
        Key in `adata.obs` to use for observations.
    genes : List[str]
        List of genes to perturb.
    subset_key : str
        Key in `adata.obs` to subset the data.
    subset_val : str
        Value in `subset_key` to subset the data.
    gene_id_key : str, optional
        Key in `adata.var` for gene identifiers. Default is "gene_ids".
    target_value : float, optional
        Target value for the perturbation. Default is 10.
    PCA : optional
        PCA object for dimensionality reduction. Default is None.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    use_key : str, optional
        Key in `adata.obsm` for the data to use. Default is "X_scaled".
    replicates : int, optional
        Number of replicates for the perturbation. Default is 5.
    N : int, optional
        Number of cells to perturb. Default is 200.
    save_simulation : bool, optional
        Whether to save the simulation results. Default is False.
    *args :
        Additional arguments.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    PerturbationExperimentResult
        The result of the perturbation experiment.
    """
    perturbation = FatePerturbationExperiment(
        seed=seed,
        use_key=use_key,
        replicates=replicates,
        N=N,
        save_simulation=save_simulation,
        *args,
        **kwargs,
    )
    return perturbation(
        adata=adata,
        model=model,
        t_sim=t_sim,
        obs_key=obs_key,
        genes=genes,
        gene_id_key=gene_id_key,
        subset_key=subset_key,
        subset_val=subset_val,
        target_value=target_value,
        PCA=PCA,
    )
