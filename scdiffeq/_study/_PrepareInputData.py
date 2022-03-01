import adata_sdk as sdk
import scdiffeq as sdq
import vinplots

from .._clonal._return_clones_present_at_all_timepoints import _return_clones_present_at_all_timepoints

class _PrepareInputData:

    """This is currently a working (un-optimized) implementation, which I expect to change!"""

    def __init__(self, adata):

        self._adata = adata
        obs = self._adata.obs.copy()

        self._test_obs = obs.loc[obs["heldout_mask"] == 0]
        self._train_obs = obs.loc[obs["heldout_mask"] == 1]

        self._test_idx = self._test_obs.index.astype(int)
        self._train_idx = self._train_obs.index.astype(int)

#         self._NM_early_d2_test = (
#             self._test_obs.loc[self._test_obs["Time point"] == 2]
#             .loc[self._test_obs["has_fate_mask"] == True]
#             .loc[self._test_obs["early_neu_mo"] == True]
#             .dropna()
#         )
#         self._NM_early_d2_train = (
#             self._train_obs.loc[self._train_obs["Time point"] == 2]
#             .loc[self._train_obs["has_fate_mask"] == True]
#             .loc[self._train_obs["early_neu_mo"] == True]
#             .dropna()
#         )

        self._NM_early_d2_test_idx = self._NM_early_d2_test.index.astype(int)
        self._NM_early_d2_train_idx = self._NM_early_d2_train.index.astype(int)

        self._train = self._adata[self._train_idx].copy()
        self._test = self._adata[self._test_idx].copy()

        self._train.obs = self._train.obs.reset_index(drop=True)
        self._test.obs = self._test.obs.reset_index(drop=True)

    def filter_incomplete_lineages(self):

        self.train = _return_clones_present_at_all_timepoints(
            self._train, plot=False
        )[0]
        self.test = _return_clones_present_at_all_timepoints(self._test, plot=False)[
            0
        ]

        for obj in [self.train, self.test]:
            obj.obs = obj.obs.reset_index(drop=True)
            obj.obs.index = obj.obs.index.astype(str)

    def write(self, destination="./", silent=False):

        sdk.write_loaded_h5ad(self.train, name="train", outpath=destination, silent=silent)
        print("\n")
        sdk.write_loaded_h5ad(self.test, name="test", outpath=destination, silent=silent)
        
def _prepare_input_data(adata, destination="./", silent=False):

    """
    Split data into test and train according to various other comparative method's protocols (e.g., Weinreb et al., 2020)
    """

    data = _PrepareInputData(adata)
    data.filter_incomplete_lineages()
    data.write(destination=destination, silent=silent)

    return data