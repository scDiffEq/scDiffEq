from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import ABCParse
import pickle
import os

from ..core import utils



class DimensionReduction(ABCParse.ABCParse):
    def __init__(
        self,
        adata,
        n_pcs=50,
        n_umap_components=2,
        train_key="train",
        test_key="test",
        write=True,
        save_path="./",
        pkl_protocol=4,
        use_saved=True,
    ):

        self.__parse__(locals(), private=[None])

        self.df = self.adata.obs.copy()
        self._INFO = utils.InfoMessage()

        self.train_idx = self.adata[self.df[self.train_key]].obs.index
        self.test_idx = self.adata[self.df[self.test_key]].obs.index
        self.unique_test_idx = self.test_idx[
            [idx not in self.train_idx for idx in self.test_idx]
        ]

        self.X_train = self.adata[self.train_idx].X.toarray()
        self.X = self.adata.X
        
        self.n_cells_train = self.X_train.shape[0]
        self.n_cells  = self.X.shape[0]

        self.SCALER_KWARGS = utils.function_kwargs(
            func=StandardScaler, kwargs=self._PARAMS
        )
        self.PCA_KWARGS = utils.function_kwargs(func=PCA, kwargs=self._PARAMS)
        self.UMAP_KWARGS = utils.function_kwargs(func=UMAP, kwargs=self._PARAMS)

    @property
    def scaler(self):
        if not hasattr(self, "_scaler_model"):
            if os.path.exists(self.path_scaler) and (not self.use_saved):
                self._scaler_model = pickle.load(open(self.path_scaler, "rb"))
            else:
                self._scaler_model = StandardScaler(**self.SCALER_KWARGS)
        return self._scaler_model

    @property
    def PCA(self):
        if not hasattr(self, "_pca_model"):
            if os.path.exists(self.path_pca) and self.use_saved:
                self._pca_model = pickle.load(open(self.path_pca, "rb"))
            else:
                self._INFO(f"Fitting PCA model on {self.n_cells_train} training cells")
                self._pca_model = PCA(n_components=self.n_pcs, **self.PCA_KWARGS)
        return self._pca_model

    @property
    def UMAP(self):
        if not hasattr(self, "_UMAP_model"):
            if os.path.exists(self.path_umap) and self.use_saved:
                self._INFO("Loading UMAP model from file")
                self._UMAP_model = pickle.load(open(self.path_umap, "rb"))
            else:
                self._UMAP_model = UMAP(
                    n_components=self.n_umap_components, **self.UMAP_KWARGS
                )
        return self._UMAP_model

    @property
    def path_scaler(self):
        return os.path.join(self.save_path, "scaler_model.pkl")

    @property
    def path_pca(self):
        return os.path.join(self.save_path, "pca_model.pkl")

    @property
    def path_umap(self):
        return os.path.join(self.save_path, "umap_model.pkl")

    def scale(self):
        self.X_scaled_train = self.scaler.fit_transform(self.X_train)
        self.X_scaled = self.scaler.transform(self.X)
        if self.write:
            pickle.dump(
                obj=self.scaler,
                file=open(self.path_scaler, "wb"),
                protocol=self.pkl_protocol,
            )

    def pca(self):
        self.X_pca_train = self.PCA.fit_transform(self.X_scaled_train)
        self._INFO(f"Transforming all cells ({self.n_cells}) using trained PCA model")
        self.X_pca = self.PCA.transform(self.X_scaled)
        if self.write:
            self._INFO(f"Saving PCA model to: {self.path_pca}")
            pickle.dump(
                obj=self.PCA,
                file=open(self.path_pca, "wb"),
                protocol=self.pkl_protocol,
            )

    def umap(self):
        if os.path.exists(self.path_umap) and self.use_saved:
            self.X_umap_train = self.UMAP.embedding_
            self._INFO(f"Transforming all cells ({self.n_cells}) using trained UMAP model")
            self.X_umap = self.UMAP.transform(self.X_pca)
        else:
            self._INFO(f"Fitting UMAP model on {self.n_cells_train} training cells")
            self.X_umap_train = self.UMAP.fit_transform(self.X_pca_train)
            self._INFO(f"Transforming all cells ({self.n_cells}) using trained UMAP model")
            self.X_umap = self.UMAP.transform(self.X_pca)
            if self.write:
                self._INFO(f"Saving UMAP model to: {self.path_umap}")
                pickle.dump(
                    obj=self.UMAP,
                    file=open(self.path_umap, "wb"),
                    protocol=self.pkl_protocol,
                )