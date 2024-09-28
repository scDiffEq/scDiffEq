# -- import packages: ---------------------------------------------------
import anndata
import logging
import numpy as np
import torch


# -- initialize logger: ----------------------------------------------------
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)


# -- import local dependencies: ------------------------------------------
from ._perturb import perturb


# -- set typing: -----------------------------------------------------------
from typing import List, Optional


# -- API-facing function: -------------------------------------------------
def perturb_scan_z_range(
    adata: anndata.AnnData,
    model: 'scdiffeq.scDiffEq',
    obs_key: str,
    subset_key: str,
    subset_val: str,
    genes: List[str],
    t_sim: torch.Tensor,
    z_range: np.ndarray,
    PCA: Optional = None,
    seed: int = 0,
    gene_id_key: str = "gene_ids",
    use_key: str = 'X_scaled',
    replicates: int = 5,
    N: int = 2000,
    save_simulation: bool = False,
    *args,
    **kwargs,
):
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
            PCA=PCA,
            *args,
            **kwargs,
        )
        logger.info(f"Perturbed {len(genes)} genes (Z={z})")
    return Perturbed
