

# -- import packages: ----------------------------------------------------------
import adata_query
import sklearn
import pickle
import umap
import os
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ..core import utils
# from ._x_use import fetch_formatted_data


# -- type setting: -------------------------------------------------------------
NoneType = type(None)


# -- controller class: ---------------------------------------------------------
class DimensionReduction(ABCParse.ABCParse):
    _SCALER_FIT, _PCA_FIT, _UMAP_FIT = [False] * 3

    def __init__(
        self,
        adata,
        use_key: str = "X",
        n_pcs: int = 50,
        n_umap_components: int = 2,
        train_key="train",
        test_key="test",
        write=True,
        save_path="./reducer_models",
        use_saved=True,
        pkl_protocol=4,
        scaler_kwargs={},
        pca_kwargs={},
        umap_kwargs={},
        *args,
        **kwargs,
    ):

        self._configure(locals())
        
    def _configure(self, kwargs):

        """
        
        * If train_key is not in adata, set all cells to train.
        """
        self.__parse__(kwargs, public=[None])
        
        if not self._train_key in self._adata.obs.columns:
            self._adata.obs[self._train_key] = True
            
        self.df = self._adata.obs.copy()
        self._INFO = utils.InfoMessage()

        if not os.path.exists(self._save_path):
            self._INFO(f"mkdir: {self._save_path}")
            os.mkdir(self._save_path)

    def _configure_kwargs(self, func, kwarg_subset_key, specific_kwarg: dict = None):
        """"""
        PARAMS = self._PARAMS
        PARAMS.update(self._PARAMS[kwarg_subset_key])
        if not isinstance(specific_kwarg, NoneType):
            PARAMS.update(specific_kwarg)
        return ABCParse.function_kwargs(func=func, kwargs=PARAMS)

    @property
    def _SCALER_MODULE(self):
        return sklearn.preprocessing.StandardScaler

    @property
    def _PCA_MODULE(self):
        return sklearn.decomposition.PCA

    @property
    def _UMAP_MODULE(self):
        return umap.UMAP

    @property
    def _SCALER_KWARGS(self):
        return self._configure_kwargs(
            func=self._SCALER_MODULE,
            kwarg_subset_key="scaler_kwargs",
        )

    @property
    def _PCA_KWARGS(self):
        return self._configure_kwargs(
            func=self._PCA_MODULE,
            kwarg_subset_key="pca_kwargs",
            specific_kwarg={"n_components": self._n_pcs},
        )

    @property
    def _UMAP_KWARGS(self):
        return self._configure_kwargs(
            func=self._UMAP_MODULE,
            kwarg_subset_key="umap_kwargs",
            specific_kwarg={"n_components": self._n_umap_components},
        )

    @property
    def train_idx(self):
        if not hasattr(self, "_train_idx"):
            self._train_idx = self._adata[self.df[self._train_key]].obs.index
        return self._train_idx

    @property
    def test_idx(self):
        if not hasattr(self, "_test_idx"):
            self._test_idx = self._adata[self.df[self._test_key]].obs.index
        return self._test_idx

    @property
    def unique_test_idx(self):
        if not hasattr(self, "_unique_test_idx"):
            train, test = self.train_idx, self.test_idx
            self._unique_test_idx = test[[idx not in train for idx in test]]
        return self._unique_test_idx

    @property
    def X_train(self):
        if not hasattr(self, "_X_train"):
            self._X_train = adata_query.fetch(
                adata=self._adata[self.train_idx],
                key=self._use_key,
                torch=False,
            )

        return self._X_train

    @property
    def X(self):
        if not hasattr(self, "_X"):
            self._X = adata_query.fetch(
                adata=self._adata,
                key=self._use_key,
                torch=False,
            )

        return self._X

    @property
    def _N_CELLS_TRAIN(self):
        return self.X_train.shape[0]

    @property
    def _N_CELLS(self):
        return self.X.shape[0]

    @property
    def _FIT_USING_ALL_CELLS(self):
        return self._N_CELLS_TRAIN == self._N_CELLS

    # -- paths to saved models: ------------------------------------------------
    @property
    def _PATH_SCALER(self):
        return os.path.join(self._save_path, "scaler_model.pkl")

    @property
    def _PATH_PCA(self):
        return os.path.join(self._save_path, "pca_model.pkl")

    @property
    def _PATH_UMAP(self):
        return os.path.join(self._save_path, "umap_model.pkl")

    def _msg(self, PATH):

        if not os.path.exists(PATH):
            self._INFO(f"PATH: {PATH} DOES NOT EXIST.")
        else:
            if self._use_saved:
                use = "Will use this model."
            else:
                use = "Not using."
            self._INFO(f"PATH: {PATH} DOES EXIST. {use}")

    def _OK_TO_LOAD(self, PATH):
        return os.path.exists(PATH) and (self._use_saved)

    def _LOAD(self, PATH, key):
        self._INFO(f"Loading {key} from: {PATH}")
        return pickle.load(open(PATH, "rb"))

    def _SETUP_MODULE(self, key):
        MODULE = getattr(self, f"_{key}_MODULE")
        KWARGS = getattr(self, f"_{key}_KWARGS")
        return MODULE(**KWARGS)

    def _CONFIGURE_MODEL(self, key):
        self._INFO(f"Configuring: {key} model")
        """try (1) loading from file if it exists then (2) building it from scratch"""

        PATH = getattr(self, f"_PATH_{key}")
        self._msg(PATH)
        if self._OK_TO_LOAD(PATH):
            setattr(self, f"_{key}_FIT", True)
            return self._LOAD(PATH, key)

        return self._SETUP_MODULE(key)
    
    # -- models: ---------------------------------------------------------------
    @property
    def SCALER(self):
        if not hasattr(self, "_SCALER_MODEL"):
            self._SCALER_MODEL = self._CONFIGURE_MODEL("SCALER")
        return self._SCALER_MODEL

    @property
    def PCA(self):
        if not hasattr(self, "_PCA_MODEL"):
            self._PCA_MODEL = self._CONFIGURE_MODEL("PCA")
        return self._PCA_MODEL

    @property
    def UMAP(self):
        if not hasattr(self, "_UMAP_MODEL"):
            self._UMAP_MODEL = self._CONFIGURE_MODEL("UMAP")
        return self._UMAP_MODEL

    def _write_model(self, MODEL, key):
        PATH = getattr(self, f"_PATH_{key}")
        self._INFO(f"Saving {key} model to: {PATH}")
        pickle.dump(MODEL, open(PATH, "wb"))

    def _fit(
        self,
        X_input,
        X_input_train,
        key,
    ):

        MODEL = getattr(self, key)
        FIT = getattr(self, f"_{key}_FIT")

        if FIT:
            self._INFO(f"Using previously fit {key} model")
            if (
                (key == "UMAP")
                and self._FIT_USING_ALL_CELLS
                and (MODEL.embedding_.shape[0] == self._N_CELLS)
            ):
                self._INFO("Using cached embedding from UMAP model")
                X_transformed_train = MODEL.embedding_
            else:
                X_transformed_train = MODEL.transform(X_input_train)
            if self._FIT_USING_ALL_CELLS:
                X_transformed = X_transformed_train
            else:
                X_transformed = MODEL.transform(X_input)
        else:
            self._INFO(f"Fitting {key} model")

            X_transformed_train = MODEL.fit_transform(X_input_train)

            if self._FIT_USING_ALL_CELLS:
                X_transformed = X_transformed_train
            else:
                X_transformed = MODEL.transform(X_input)

            self._write_model(MODEL, key)

        return MODEL, X_transformed, X_transformed_train

    # -- fit functions: --------------------------------------------------------
    def fit_scaler(self):

        self._SCALER_MODEL, self.X_scaled, self.X_scaled_train = self._fit(
            X_input=self.X,
            X_input_train=self.X_train,
            key="SCALER",
        )
        self._adata.layers["X_scaled"] = self.X_scaled

    def fit_pca(self):

        self._PCA_MODEL, self.X_pca, self.X_pca_train = self._fit(
            X_input=self.X_scaled,
            X_input_train=self.X_scaled_train,
            key="PCA",
        )
        self._adata.obsm["X_pca"] = self.X_pca

    def fit_umap(self):

        self._UMAP_MODEL, self.X_umap, self.X_umap_train = self._fit(
            X_input=self.X_pca,
            X_input_train=self.X_pca_train,
            key="UMAP",
        )
        self._adata.obsm["X_umap"] = self.X_umap
