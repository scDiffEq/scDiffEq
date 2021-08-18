import os
import vintools as v

from ._save_adata_uns import _save_adata_uns
from ._create_outs_structure import _create_outs_structure
from ..._utilities._zip_results_archive import _zip_results_archive


def _save(
    self,
    outdir=os.getcwd(),
    pickle_dump_list=["pca", "loss"],
    pass_keys=["split_data", "data_split_keys", "RunningAverageMeter"],
    put_back=False,
):

    """
    Save current scDiffEq state.

    Parameters:
    -----------

    Returns:
    --------

    Notes:
    ------
    (1) creates the following structure:
        [/path/]/scdiffeq_outs/
        │
        ├── simulation_figures/
        ├── model_checkpoints/
        ├── results_figures/
        └── adata
            ├── uns/
            └── adata.h5ad

    (2) Operates over n steps.
    (1) saves essential components of the DiffEq class.
    (2) saves AnnData in multiple steps
    (3) saves any optional peripherals.
    (4) zip files and move from tempdir to exportdir (optional)
    """

    _create_outs_structure(self)
    backup_adata = self.adata
    _save_adata_uns(
        self, pickle_dump_list=pickle_dump_list, pass_keys=pass_keys,
    )
    h5ad_outpath = os.path.join(
        self._AnnData_path, "epoch_{}.adata.h5ad".format(self.epoch)
    )
    self.adata.write_h5ad(h5ad_outpath)
    _zip_results_archive(self._outs_path)
    if put_back:
        self.adata = backup_adata
