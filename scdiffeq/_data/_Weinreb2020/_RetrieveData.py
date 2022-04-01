def _return_early_neu_mo(df, early_neu_mo_mask_key, early_neu_mo_mask):
    return df.loc[df[early_neu_mo_mask_key] == early_neu_mo_mask]


def _return_time(df, time_key, time):
    return df.loc[df[time_key] == time]


def _return_neu_mo(df, neu_mo_mask_key, neu_mo_mask=True):
    return df.loc[df[neu_mo_mask_key] == neu_mo_mask]


def _return_heldout(df, heldout_mask_key, heldout_mask):
    return df.loc[df[heldout_mask_key] == heldout_mask]


def _return_fate_mask(df, fate_mask_key, fate_mask):
    return df.loc[df[fate_mask_key] == fate_mask]

def _return_lineage_traced(df, lineage_key):
    return df.loc[df[lineage_key].dropna().index]

class _RetrieveData:
    def __init__(
        self,
        adata,
        time_key="Time point",
        neu_mo_mask_key="neu_mo_mask",
        heldout_mask_key="heldout_mask",
        fate_mask_key="has_fate_mask",
        early_neu_mo_mask_key="early_neu_mo",
        lineage_key="clone_idx",
        verbose=False
    ):

        """"""

        self._df = adata.obs.copy()
        self._verbose = verbose
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))
        
        self._time_key = time_key
        self._neu_mo_mask_key = neu_mo_mask_key
        self._heldout_mask_key = heldout_mask_key
        self._fate_mask_key = fate_mask_key
        self._early_neu_mo_mask_key = early_neu_mo_mask_key
        self._lineage_key = lineage_key

    def time(self, time):

        self._df = _return_time(self._df, self._time_key, time)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def neu_mo(self, neu_mo_mask=True):

        self._df = _return_neu_mo(self._df, self._neu_mo_mask_key, neu_mo_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def early_neu_mo(self, early_neu_mo_mask=1):
        self._df = _return_early_neu_mo(
            self._df, self._early_neu_mo_mask_key, early_neu_mo_mask
        )
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def heldout(self, heldout_mask=0):

        self._df = _return_heldout(self._df, self._heldout_mask_key, heldout_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def fate_mask(self, fate_mask=1):
        self._df = _return_fate_mask(self._df, self._fate_mask_key, fate_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def neu_mo_test_set_early(self, early_neu_mo_mask=1, heldout_mask=0, fate_mask=1):

        """"""

        self._df = _return_early_neu_mo(
            self._df, self._early_neu_mo_mask_key, early_neu_mo_mask
        )
        self._df = _return_heldout(self._df, self._heldout_mask_key, heldout_mask)
        self._df = _return_fate_mask(self._df, self._fate_mask_key, fate_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def neu_mo_d2_all(self, time=2, neu_mo_mask=True):

        """"""

        self._df = _return_time(self._df, self._time_key, time)
        self._df = _return_neu_mo(self._df, self._neu_mo_mask_key, neu_mo_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))

    def neu_mo_test_set_all(self, heldout_mask=0, fate_mask=1):

        """"""

        self._df = _return_heldout(self._df, self._heldout_mask_key, heldout_mask)
        self._df = _return_fate_mask(self._df, self._fate_mask_key, fate_mask)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))
        
    def lineage_traced(self):
        
        self._df = _return_lineage_traced(self._df, self._lineage_key)
        if self._verbose:
            print("Dataset contains: {} cells.".format(self._df.shape[0]))