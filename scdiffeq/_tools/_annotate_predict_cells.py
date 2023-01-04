
import numpy as np


class Annotator:
    def __init__(self, adata):
        self.adata = adata

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

    def message(self):
        print(
            "{} cells labelled under adata.obs['{}']".format(
                self.n_cells_labelled, self.key
            )
        )

    def __call__(self, idx, key):
        self.key = key
        self.set_true(idx)
        self.adata.obs[key] = self.col
        self.message()


# -- API-facing function: ------------------------------------------------------------
def annotate_predict_cells(adata, idx, key="predict"):
    """
    Given an index, create a boolean vector to be passed to model.predict(adata)
    """
    annot = Annotator(adata)
    annot(idx=idx, key=key)
