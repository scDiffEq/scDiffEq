
# import packages #
# --------------- #
import anndata as a
import cell_tools as cell
import matplotlib.pyplot as plt
import os


# local imports #
# ------------- #
from ._io._download_preprocessed_data import _download_preprocessed_anndata_from_GCP
from ._io._download_preprocessed_data import _list_downloaded_files
from ._io._read_downloaded_data import _read_downloaded_data

from ._analyses._Weinreb2020_Figure5_Annotations import _annotate_adata_with_Weinreb2020_Fig5_predictions

from ._preprocessing._Weinreb2020_PathDict import _Weinreb2020_PathDict
from ._preprocessing._return_PRESCIENT_cell_cycle_gene_set import _return_PRESCIENT_cell_cycle_gene_set
from ._preprocessing._read_Weinreb2020_inputs_to_AnnData import _read_Weinreb2020_inputs_to_AnnData
from ._preprocessing._write_adata import _write_adata
from ._preprocessing._plot_dataset import _plot_dataset
from ._preprocessing._annotate_clonal_barcodes import _annotate_clonal_barcodes
from ._preprocessing._ClonalAnnData import _ClonalAnnData

class _Weinreb2020_Dataset:
    def __init__(
        self,
        destination_path="./scdiffeq_data/Weinreb2020_preprocessed/",
        preprocessed_data_bucket="scdiffeq-data/Weinreb2020/preprocessed_adata/*",
        verbose=True,
    ):

        """
        Parameters:
        -----------
        destination_path
            destination path for downloaded files.
            default: './scdiffeq_data/'
            type: str

        bucket_path
            path to stored data in GCP.
            default: 'scdiffeq-data/Weinreb2020/preprocessed_adata/*'
            type: str

        force
            toggle force-redownload of files from GCP.
            default: False
            type: bool

        verbose
            toggle messaging
            default: True
            type: bool

        Returns:
        --------
        None, instantiates class.

        Notes:
        ------
        (1) No required arguments to instantiate the class.
        """

        self._verbose = True
        self._downloaded_files = False
        self._destination_path = destination_path
        self._preprocessed_data_bucket = preprocessed_data_bucket
        self._CytoTRACE_df_path = os.path.join(
            self._destination_path, "LARRY.CytoTRACE.DataFrame.csv"
        )
        self._just_downloaded = False

    def download_preprocessed(self, force=False):
        
        self._force = force
        self._downloaded_files = _download_preprocessed_anndata_from_GCP(
            self._destination_path, self._preprocessed_data_bucket, force, self._verbose
        )
        self._just_downloaded = True

    def read_preprocessed(self, return_adata=False, force_redownload=False):
        
        self._force = force_redownload
        
        # this step at least lists what's available in case something goes wrong in the next step.
        if not self._downloaded_files:
            self._downloaded_files = _list_downloaded_files(self._destination_path,
                                                        self._verbose,
                                                        after_download=self._just_downloaded)
        
        if not self._downloaded_files:
            self._downloaded_files = _download_preprocessed_anndata_from_GCP(
                self._destination_path,
                self._preprocessed_data_bucket,
                self._force,
                self._verbose,
            )
        self._adata = _read_downloaded_data(
            self._downloaded_files, self._CytoTRACE_df_path, verbose=self._verbose
        )
        if return_adata:
            return self._adata
    
    def annotate_predictions(self):
        
        self._adata = _annotate_adata_with_Weinreb2020_Fig5_predictions(self._adata)
    
    class Preprocessing:

        """A sub-class of the Weinreb2020 dataset where preprocessing can be recapitulated."""

        def __init__(self, path=False, write_path="Weinreb2020.adata.h5ad"):

            """"""

            self._adata = False
            self._write_path = write_path
            if path:
                self._data_dir = path
                self._PathDict = _Weinreb2020_PathDict(path)
                self._adata = _read_Weinreb2020_inputs_to_AnnData(self._PathDict)

            elif os.path.exists(self._write_path):
                self._adata = a.read_h5ad(self._write_path)

            else:
                print("Pass a path to Weinreb 2020 inputs or preprocessed adata.")

            if self._adata:
                print(self._adata)

        def filtering(
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

                self._adata = cell.rna.remove_correlated_genes(
                    self._adata, signature_genes=cell_cycle_genes
                )
                self._adata = self._adata[
                    :, :2447
                ].copy()  # unsure why, but the PRESCIENT authors do this step
                self._adata.uns["highly_variable_genes_idx"] = self._adata.uns[
                    "highly_variable_genes_idx"
                ][:2447]

            self._adata.obs = _annotate_clonal_barcodes(self._adata)

        def dimension_reduction(
            self,
            pca_components=50,
            umap_components=2,
            umap_metric="euclidean",
            umap_verbosity=False,
            plot=True,
            figsize=1.5,
            plot_savedir="./",
        ):
            cell.tl.pca(self._adata, n_components=pca_components)
            cell.tl.umap(
                self._adata,
                n_components=umap_components,
                metric=umap_metric,
                verbose=umap_verbosity,
            )

            if plot:
                _plot_dataset(self._adata, figsize, save=plot_savedir)

        def write(
            self,
            write_path=False,
        ):
            if write_path:
                self._write_path = write_path

            _write_adata(self._adata, path=self._write_path)

        def format_clones(self, annot_dir="./"):

            self._clonal = _ClonalAnnData(self._adata, annot_dir)
            self._clonal_adata = self.clonal.load()
            self._clonal.prepare_for_training()


def _load_preprocessed_Weinreb2020_Dataset(
    destination_path="./scdiffeq_data/Weinreb2020_preprocessed/",
    preprocessed_data_bucket="scdiffeq-data/Weinreb2020/preprocessed_adata/*",
    force_redownload=False,
    verbose=True,
):

    """"""

    Weinreb2020 = _Weinreb2020_Dataset(
        destination_path, preprocessed_data_bucket, verbose
    )
    Weinreb2020.download_preprocessed(force=force_redownload)
    Weinreb2020.read_preprocessed()
    Weinreb2020.annotate_predictions()
    
    if verbose:
        print("\n{}".format(Weinreb2020._adata))
    
    return Weinreb2020._adata
    