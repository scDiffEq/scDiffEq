
# -- import packages: ---------------------------------------------------------
import anndata
import torch


# -- import local dependencies: -----------------------------------------------
from ._fate_perturbation_experiment import FatePerturbationExperiment


# -- set typing: --------------------------------------------------------------
from typing import List, Optional


# -- API-facing function: -----------------------------------------------------
def perturb(
    adata: anndata.AnnData,
    model: 'scdiffeq.scDiffEq',
    t_sim: torch.Tensor,
    obs_key: str,
    genes: List[str],
    subset_key: str,
    subset_val: str,
    gene_id_key: str = "gene_ids",
    target_value: float = 10,
    PCA: Optional = None,
    seed: int = 0,
    use_key: str = 'X_scaled',
    replicates: int = 5,
    N: int = 200,
    save_simulation: bool = False,
    *args,
    **kwargs,
):
    """
    
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
