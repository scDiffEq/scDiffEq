
from ._RetrieveData import _RetrieveData

def _return_early_neu_mo_test(adata):

    """"""

    data = _RetrieveData(adata)
    data.neu_mo_test_set_early()
    idx = data._df.index

    return adata[idx].copy()


class _TestSet:
    def __init__(self, adata):

        """

        Parameters:
        -----------
        adata

        TestSet.adata
        TestSet.clones
        """

        self.adata = _return_early_neu_mo_test(adata)
        self.lineages = self.adata.obs["clone_idx"].unique()