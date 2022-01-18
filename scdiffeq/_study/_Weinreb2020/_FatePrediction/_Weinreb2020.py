
# import packages #
# --------------- #
import anndata as a
import cell_tools as cell
import os


# local imports #
# ------------- #
from ._supporting_functions._Weinreb2020_PathDict import _Weinreb2020_PathDict
from ._supporting_functions._return_PRESCIENT_cell_cycle_gene_set import _return_PRESCIENT_cell_cycle_gene_set
from ._supporting_functions._read_Weinreb2020_inputs_to_AnnData import _read_Weinreb2020_inputs_to_AnnData
from ._supporting_functions._return_PRESCIENT_cell_cycle_gene_set import _return_PRESCIENT_cell_cycle_gene_set
from ._supporting_functions._write_adata import _write_adata
from ._supporting_functions._plot_dataset import _plot_dataset
from ._supporting_functions._annotate_clonal_barcodes import _annotate_clonal_barcodes
from ._supporting_functions._ClonalAnnData import _ClonalAnnData


class _Weinreb2020:
    def __init__(self, path=False, write_path = "Weinreb2020.adata.h5ad"):

        """"""
        
        self.adata = False
        self._write_path = write_path
        if path:
            self._data_dir = path
            self._PathDict = _Weinreb2020_PathDict(path)
            self.adata = _read_Weinreb2020_inputs_to_AnnData(self._PathDict)
            
        elif os.path.exists(self._write_path):
            self.adata = a.read_h5ad(self._write_path)
            
        else:
            print("Pass a path to Weinreb 2020 inputs or preprocessed adata.")
        
        if self.adata:
            print(self.adata)

    def preprocess(
        self,
        cell_cycle_additions=False,
        base_idx=[],
        min_var_score_percentile=85,
        min_counts=3,
        min_cells=3,
        plot=True,
        sample_name="Variable genes",
        return_hv_genes=False,
        filter_features=True,
    ):
        
        if filter_features:
            cell_cycle_genes = _return_PRESCIENT_cell_cycle_gene_set(
                add=cell_cycle_additions
            )

            cell.rna.filter_static_genes(
                self.adata,
                base_idx,
                min_var_score_percentile,
                min_counts,
                min_cells,
                plot,
                sample_name,
                return_hv_genes,
            )

            self.adata = cell.rna.remove_correlated_genes(
                self.adata, signature_genes=cell_cycle_genes
            )
            self.adata = self.adata[
                :, :2447
            ].copy()  # unsure why, but the PRESCIENT authors do this step
            self.adata.uns["highly_variable_genes_idx"] = self.adata.uns[
                "highly_variable_genes_idx"
            ][:2447]
            
        self.adata.obs = _annotate_clonal_barcodes(self.adata)
        

    def dimension_reduction(
        self,
        pca_components=50,
        umap_components=2,
        umap_metric="euclidean",
        umap_verbosity=True,
        plot=True,
        figsize=1.5,
        plot_savedir="./",
    ):
        cell.tl.pca(self.adata, n_components=pca_components)
        cell.tl.umap(
            self.adata,
            n_components=umap_components,
            metric=umap_metric,
            verbose=umap_verbosity,
        )
        
        if plot:
            _plot_dataset(self.adata, figsize, save=plot_savedir)
            
    def write(
        self,
        write_path=False,
    ):
        if write_path:
            self._write_path = write_path

        _write_adata(self.adata, path=self._write_path)
        
        
    def format_clones(self, annot_dir="./"):
        
        self.clonal = _ClonalAnnData(self.adata, annot_dir)
        self.clonal_adata = self.clonal.load()
        self.clonal.prepare_for_training()