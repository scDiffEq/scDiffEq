
from .._utilities import Base
import scdiffeq as sdq
import umap


class UMAP(Base):
    def __init__(
        self,
        use_key="X_pca",
        n_neighbors=15,
        metric="euclidean",
        spread=2,
        **kwargs,
    ):
        
        super(UMAP, self).__init__()

        self.__parse__(locals())
        self.__config__()

    def __config__(self):

        UMAP_KWARGS = sdq._core.utils.extract_func_kwargs(umap.UMAP, self._KWARGS)
        self.model = umap.UMAP(**UMAP_KWARGS)

    def forward(self, adata):

        X_use = adata.obsm[self.use_key]
        n_cells = X_use.shape[0]
        print("Fitting UMAP model to {} cells".format(n_cells))
        return self.model.fit_transform(X_use)

    def __call__(self, adata, subset_key=None):

        self.__parse__(locals(), ignore=["adata"])
        return self.forward(adata)