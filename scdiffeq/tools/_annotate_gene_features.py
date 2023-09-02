
# -- import packages: ----------------------------------------------------------
import sklearn
import anndata
import ABCParse

# -- import local dependencies: ------------------------------------------------
from ..core import utils


# -- set typing: ---------------------------------------------------------------
from typing import Optional


# -- controller class: ---------------------------------------------------------
class GeneCompatibility(ABCParse.ABCParse):

    """
    A simulation is produced in a latent space. To study gene-level features, we
    can modify the simulated adata to be annotated with gene-level features (which
    otherwise only carries latent space features).
    """

    def __init__(
        self,
        gene_id_key="gene_ids",
        PCA: Optional[sklearn.decomposition.PCA] = None,
        key_added: str = "X_gene",
    ):

        self.__parse__(locals(), public=[None])
        self._INFO = utils.InfoMessage()

    def _format_var_names(self):
        self.adata_sim.uns[self._gene_id_key] = self.adata.var[self._gene_id_key]
        self._INFO(f"Gene names added to: `adata_sim.uns['{self._gene_id_key}']`")

    def _format_inverted_expression(self):

        assert not self._PCA is None, "Must supply PCA model!"

        self.adata_sim.obsm[self._key_added] = self._PCA.inverse_transform(
            self.adata_sim.X
        )
        self._INFO(f"Inverted expression added to: `adata_sim.obsm['{self._key_added}']`")

    def __call__(
        self, adata: anndata.AnnData, adata_sim: anndata.AnnData, *args, **kwargs
    ):

        self.__update__(locals())

        self._format_var_names()
        if not self._PCA is None:
            self._format_inverted_expression()

# -- API-facing function: ------------------------------------------------------
def annotate_gene_features(
    adata_sim: anndata.AnnData,
    adata: anndata.AnnData,
    PCA: Optional[sklearn.decomposition.PCA] = None,
    gene_id_key="gene_ids",
    key_added: str = "X_gene",
):
    """
    Annotate simulation with gene-level features

    Parameters
    ----------
    adata_sim: anndata.AnnData
        Simulated AnnData object

    adata: anndata.AnnData
        Reference AnnData object

    PCA: Optional[sklearn.decomposition.PCA]
        default: None

    gene_id_key: str
        default: "gene_ids"

    key_added: str
        default: "X_gene"

    Returns
    -------
    None, modifies adata_sim in-place.
    """
    gene_compatibility = GeneCompatibility(
        gene_id_key=gene_id_key, PCA=PCA, key_added=key_added
    )
    return gene_compatibility(adata_sim=adata_sim, adata=adata)
