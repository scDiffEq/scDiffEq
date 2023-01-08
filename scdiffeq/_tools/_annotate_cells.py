
import numpy as np


class Annotator:
    def __init__(self, adata):
        self.adata = adata
        self.df = self.adata.obs.copy()

    @property
    def n_cells(self):
        return len(self.adata)

    @property
    def empty(self):
        return np.zeros(self.n_cells, dtype=bool)

    def set_true(self, idx):
        self.col = self.empty.copy()
        self.col[idx.astype(int)] = True

    @property
    def n_cells_labelled(self):
        return self.col.sum()
    
    @property
    def obs_cols(self):
        return self.df.columns.tolist()
    
    def new_col(self, key):
        """# Make a new column when one has the identical name..."""
        if key in self.obs_cols:
            cols = [col.split(".")[0] for col in self.df.columns]
            new_key = "{}.{}".format(key, cols.count(key))
        return new_key

    def message(self):
        print(
            "{} cells labelled under adata.obs['{}']".format(
                self.n_cells_labelled, self.key
            )
        )

    def __call__(self, idx, key, verbose=False):
        self.key = key
        self.set_true(idx)
        self.adata.obs[key] = self.col
        if verbose:
            self.message()


# -- API-facing function: ------------------------------------------------------------
def annotate_cells(adata, idx, key):
    """
    Given an index, create a boolean vector to be passed to model.predict(adata)
    """
    annot = Annotator(adata)
    annot(idx=idx, key=key)
