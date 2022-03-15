
__module_name__ = "_Dataset.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice
import os

# import local functions #
# ---------------------- #
from ._download import _download_LARRY_files_from_GitHub
from ._format import _convert_mtx_to_npz
from ._io import _read_files_downloaded_from_GitHub
from ._io import _to_adata


class _AllonKleinLab_GitHub_LARRY_Dataset:
    def __init__(self, verbose=True, force=False):

        """Initialize class with a few set parameters / paths."""

        self._verbose = True
        self._force = False
        self._base_path = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro"
        self._GitHub_repo = "https://github.com/AllonKleinLab/paper-data/blob/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation/"
        self._file_basenames = [
            "normed_counts.mtx.gz",
            "gene_names.txt.gz",
            "clone_matrix.mtx.gz",
            "metadata.txt.gz",
            "neutrophil_pseudotime.txt.gz",
            "neutrophil_monocyte_trajectory.txt.gz",
        ]

    def download(self, destination=os.getcwd()):

        """Download files from GitHub"""
        
        self._destination = destination

        self._downloaded_files = _download_LARRY_files_from_GitHub(
            self._base_path,
            self._GitHub_repo,
            self._file_basenames,
            self._destination,
            verbose=self._verbose,
            force=self._force,
        )

    def convert_to_npz(self, silent=False):

        """Convert .mtx to .npz for faster loading"""

        self._ConvertedDict = _convert_mtx_to_npz(
            self._downloaded_files, self._destination, silent
        )

    def read_data(self):

        """Read data into memory"""

        self._DataDict = _read_files_downloaded_from_GitHub(
            self._downloaded_files, self._ConvertedDict, self._destination
        )

    def to_adata(
        self,
        write=True,
        h5ad_outpath="adata.LARRY.KleinLab_GitHub.h5ad",
        return_adata=False,
        silent=False,
    ):

        """
        Convert to AnnData.

        Parameters:
        -----------
        write
            default: True
            type: bool

        h5ad_outpath:
            default: 'adata.LARRY.KleinLab_GitHub.h5ad'
            type: str

        return_adata
            default: False
            type: bool

        silent
            default: False
            type: bool

        Returns:
        --------
        self._adata [ optional ]
        """

        self._adata = _to_adata(self._DataDict)

        if write:
            h5ad_outpath = os.path.join(os.getcwd(), h5ad_outpath)
            self._adata.write_h5ad(h5ad_outpath)
            if not silent:
                print(
                    "\n{} {}\n".format(
                        licorice.font_format("Writing to:", ["BOLD"]), h5ad_outpath
                    )
                )

        print(self._adata)

        if return_adata:
            return self._adata
        

def _Weinreb2020_AllonKleinLab_GitHub(
    destination=False,
    silent=False,
    verbose=True,
    force=False,
    write=True,
    h5ad_outpath="adata.Weinreb2020.AllonKleinLab_GitHub.h5ad",
    return_adata=True,
):

    """
    Download and/or load the LARRY dataset as AnnData from the Klein Lab GitHub Repo
    
    Parameters:
    -----------
    destination
        default: False
        type: bool
        
    silent
        default: False
        type: bool
        
    verbose
        default: True
        type: bool
        
    force
        default: False
        type: bool
        
    write
        default: True
        type: bool
        
    h5ad_outpath
        default: 'adata.Weinreb2020.AllonKleinLab_GitHub.h5ad'
        type: str
        
    return_adata
        default: False
        type: bool
        

    Returns:
    --------
    adata
        type: anndata
    
    Notes:
    ------
    
    """
    
    if not destination:
        destination = os.getcwd()

    data = _AllonKleinLab_GitHub_LARRY_Dataset(verbose, force)
    data.download(destination)
    data.convert_to_npz(silent)
    data.read_data()
    
    return data.to_adata(write, h5ad_outpath, return_adata, silent)