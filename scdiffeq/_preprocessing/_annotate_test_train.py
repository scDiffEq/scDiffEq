
import numpy as np

from .._data._Weinreb2020._TestSet import _TestSet

def _annotate_test_train(adata, lineage_key):

    """"""

    test = _TestSet(adata)
    test_lineages = test.lineages.tolist()

    adata.obs["test"] = adata.obs[lineage_key].astype(float).isin(test_lineages)
    adata.obs["train"] = np.invert(adata.obs["test"])
    traintest = np.full(len(adata), "train")
    traintest[np.where(adata.obs["test"])] = "test"
    adata.obs["train_test"] = traintest