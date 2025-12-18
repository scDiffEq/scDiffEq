# -- import packages: ---------------------------------------------------------
import anndata
import logging
import numpy as np
import sklearn.decomposition
import statsmodels.stats.multitest

# -- import local dependencies: -----------------------------------------------
from ._knn import kNN

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)

# -- operational cls: ---------------------------------------------------------
class ManifoldOutlierDetection:
    def __init__(
        self, adata: anndata.AnnData,
        PCA: sklearn.decomposition.PCA,
        pca_key: str = "X_pca",
        k: int = 5,
        fdr_threshold: float = 0.05,

    ):
        self.adata = adata
        self._PCA = PCA
        self._k = k
        self._fdr_threshold = fdr_threshold
        self._pca_key = pca_key

        self._fit_kNN()
        self._ref_k_dist = self._compute_mean_reference_distances()

    def _fit_kNN(self):
        self.knn = kNN(self.adata, use_key=self._pca_key, n_neighbors=self._k)

    def _compute_mean_reference_distances(self):
        self._X_query = self.adata.obsm[self._pca_key]
        _, distances = self.knn.query(X_query=self._X_query, include_distances=True)
        dist_mean = distances.mean(1)
        self.adata.obs[f'mean_dist_k_{self._k}'] = dist_mean
        return dist_mean
    
    def _compute_simulation_distances(self, adata_sim: anndata.AnnData):
        _, distances = self.knn.query(X_query=adata_sim.X, include_distances=True)
        return distances.mean(1)
    
    def _compute_empirical_p_values(self, ref_k_dist: np.ndarray, sim_k_dist: np.ndarray):
        """The empirical p-value is the fraction of distances in the
        reference set greater than or equal to the distance we observe,
        for each simulated cell to its five nearest neighbors in the
        reference set.
        """
        return np.array([(ref_k_dist >= dist).mean() for dist in sim_k_dist])
    
    def _report_on_outliers(self, rejected_cells: np.ndarray):
        if rejected_cells.sum() > 0:
            logger.info(f"Detected {rejected_cells.sum()} outliers out of {len(rejected_cells)} test cells (FDR < {self._fdr_threshold}).")
        else:
            logger.debug(f"No outliers (of {len(rejected_cells)} cells) detected.")
    
    def _update_adata(
            self,
            adata_sim: anndata.AnnData,
            rejected_cells: np.ndarray,
            empirical_p_values: np.ndarray,
            corrected_p_values: np.ndarray,
            
        ):
        adata_sim.obs[f'mean_dist_k_{self._k}'] = self._sim_k_dist
        adata_sim.obs["outlier_pval"] = empirical_p_values
        adata_sim.obs["outlier_qval"] = corrected_p_values
        adata_sim.obs["outlier_flag_fdr"] = rejected_cells
        adata_sim.uns["outlier_detection"] = {
            "fdr_threshold": self._fdr_threshold,
            "k": self._k,
            "n_outliers": rejected_cells.sum(),
            "n_total": len(rejected_cells),
        }

    def forward(self, adata_sim: anndata.AnnData):
        """Detects outliers in adata_sim based on the adata (reference)
        manifold.

        Args:
            adata_sim (anndata.AnnData): The simulation data.

        Returns:
            np.ndarray: The indices of the rejected cells.
            np.ndarray: The corrected p-values.
        """
        
        self._sim_k_dist = self._compute_simulation_distances(adata_sim=adata_sim)
        self._empirical_p_values = self._compute_empirical_p_values(
            ref_k_dist=self._ref_k_dist,
            sim_k_dist=self._sim_k_dist,
        )
        rejected_cells, corrected_p_values = statsmodels.stats.multitest.fdrcorrection(
            self._empirical_p_values,
            alpha=self._fdr_threshold,
        )
        self._report_on_outliers(rejected_cells=rejected_cells)
        self._update_adata(
            adata_sim=adata_sim,
            rejected_cells=rejected_cells,
            empirical_p_values=self._empirical_p_values,
            corrected_p_values=corrected_p_values,
        )
        return rejected_cells, corrected_p_values
    
    def __call__(self, adata_sim: anndata.AnnData):
        return self.forward(adata_sim=adata_sim)

# -- function: ----------------------------------------------------------------
def detect_manifold_outliers(
        adata_sim: anndata.AnnData,
        adata: anndata.AnnData,
        PCA: sklearn.decomposition.PCA,
        pca_key: str = "X_pca",
        k: int = 5,
        fdr_threshold: float = 0.05,
):
    
    outlier_detection = ManifoldOutlierDetection(
        adata=adata,
        PCA=PCA,
        pca_key=pca_key,
        k=k,
        fdr_threshold=fdr_threshold,
    )
    return outlier_detection(adata_sim=adata_sim)