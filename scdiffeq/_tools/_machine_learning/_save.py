
import os
import vintools as v

from ._save_adata_uns import _save_adata_uns

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
        ├── model_checkpoints/
        ├── AnnData
            ├── uns/
            ├── adata.h5ad

    (2) Operates over n steps.

    (1) saves essential components of the DiffEq class.
    (2) saves AnnData in multiple steps
    (3) saves any optional peripherals.
    (4) zip files and move from tempdir to exportdir (optional)
    """    
    outs_path = os.path.join(outdir, "scdiffeq_outs")
    AnnData_path = os.path.join(outdir, "scdiffeq_outs/AnnData/")
    uns_path = os.path.join(outdir, "scdiffeq_outs/AnnData/uns")
    
    v.ut.mkdir_flex(outdir)
    v.ut.mkdir_flex(outs_path)
    v.ut.mkdir_flex(AnnData_path)
    v.ut.mkdir_flex(uns_path)
    backup_adata = self.adata
    _save_adata_uns(
        self,
        parent_dir=outs_path,
        uns_path=uns_path,
        pickle_dump_list=pickle_dump_list,
        pass_keys=pass_keys,
    )
    h5ad_outpath = os.path.join(AnnData_path, "epoch_{}.adata.h5ad".format(self.epoch))
    self.adata.write_h5ad(h5ad_outpath)
    if put_back:
        self.adata = backup_adata
    
    print(
    """
    Saved with the following structure:\n
    {}/scdiffeq_outs/
    ├── model_checkpoints/
    ├── adata
        ├── uns/
        ├── adata.h5ad
    """.format(outdir)
    )