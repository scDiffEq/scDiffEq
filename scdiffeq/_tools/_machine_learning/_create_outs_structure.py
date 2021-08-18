# package imports #
# --------------- #

import vintools as v
import os


def _create_outs_structure(self):

    """"""

    try:
        self.outdir
    except:
        self.outdir = os.getcwd()

    self._outs_path = os.path.join(self.outdir, "scdiffeq_outs")
    self._imgs_path = os.path.join(self.outdir, "scdiffeq_outs/results_figures/")
    self._AnnData_path = os.path.join(self.outdir, "scdiffeq_outs/AnnData/")
    self._uns_path = os.path.join(self.outdir, "scdiffeq_outs/AnnData/uns")

    v.ut.mkdir_flex(self.outdir)
    v.ut.mkdir_flex(self._outs_path)
    v.ut.mkdir_flex(self._imgs_path)
    v.ut.mkdir_flex(self._AnnData_path)
    v.ut.mkdir_flex(self._uns_path)

    print(
        """
    Saved with the following structure:\n
    {}scdiffeq_outs/
    │
    ├── model_checkpoints/
    ├── results_figures/
    └── adata
        ├── uns/
        └── adata.h5ad
    """.format(
            self.outdir
        )
    )
