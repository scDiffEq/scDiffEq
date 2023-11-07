
import tqdm
import ABCParse
from ..core import utils
from ._knn import kNN
import adata_query

NoneType = type(None)


class kNNSmoothing(ABCParse.ABCParse):
    def __init__(
        self,
        adata,
        kNN=None,
        n_iters: int = 3,
        n_neighbors: int = 20,
        use_key="X_pca",
        use_tqdm=False,
    ):

        self.__parse__(locals(), public=[None])
        self._STDEV = []

    @property
    def kNN(self):
        if isinstance(self._kNN, NoneType):
            self._kNN = kNN(
                self._adata,
                use_key=self._use_key,
                n_neighbors=self._n_neighbors,
            )
        return self._kNN

    @property
    def X_use(self):
        if not hasattr(self, "_X_use"):
            self._X_use = adata_query.fetch(
                self._adata, key=self._use_key, torch=False,
            )
        return self._X_use

    def forward(self, SCORE):
        X_nn = self.kNN.query(X_query=self.X_use)
        _SCORE = SCORE[X_nn]
        self._STDEV.append(_SCORE.std(1).mean())
        return _SCORE.mean(1)
    
    def _configure_progress_bar(self):
        if self._use_tqdm:
            return tqdm.notebook.tqdm(range(self._n_iters))
        return range(self._n_iters)
    
    @property
    def _PROGRESS_BAR(self):
        if not hasattr(self, "_progress_bar"):
            self._progress_bar = self._configure_progress_bar()
        return self._progress_bar

    @property
    def SCORE_INIT(self):
        return self._adata.obs[self._key].copy().values

    def __call__(self, key: str, add_to_adata=False):

        self._key = key

        SCORE = self.SCORE_INIT
        for i in self._PROGRESS_BAR:
            SCORE = self.forward(SCORE)

        if not add_to_adata:
            return SCORE
        else:
            self._adata.obs[f"{key}_smoothed"] = SCORE