
# -- import packages: -----------------------------------
from torch_adata import LightningAnnDataModule
import os


# -- import local dependencies: -------------------------
from .. import utils


class LightningData(LightningAnnDataModule, utils.AutoParseBase):
    def __init__(
        self,
        adata=None,
        h5ad_path=None,
        batch_size=2000,
        num_workers=os.cpu_count(),
        train_val_split=[0.8, 0.2],
        use_key="X_pca",
        obs_keys=[],
        weight_key='W',
        groupby="Time point",  # TODO: make optional
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        silent=True,
        shuffle_time_labels = False,
        **kwargs,
    ):
        super(LightningData, self).__init__()
        
                    
        self.__parse__(locals(), public=[None])
        self._format_sinkhorn_weight_key()
        self._format_train_test_exposed_data()
        self.configure_train_val_split()
        

    @property
    def n_dim(self):
        if not hasattr(self, "_n_dim"):
            self._n_dim = self.train_dataset.X.shape[-1]
        return self._n_dim
    
#     def shuffle_t(self):
        
#         df = self._adata.obs.copy()
#         non_t0 = df.loc[df['t'] != 0]['t']
        
#         shuffled_t = np.zeros(len(df))
#         shuffled_t[non_t0.index.astype(int)] = np.random.choice(non_t0.values, len(non_t0))
#         self._adata.obs["t"] = shuffled_t
    
    def _format_sinkhorn_weight_key(self):
        if not self._weight_key in self._adata.obs.columns:
            self._adata.obs[self._weight_key] = 1
        self._obs_keys.append(self._weight_key)
    
    def _format_train_test_exposed_data(self):
        
        if not self._train_key in self._adata.obs.columns:
            self._adata.obs[self._train_key] = True
        if not self._test_key in self._adata.obs.columns:
            self._adata.obs[self._test_key] = False
            
    def prepare_data(self):
        ...

    def setup(self, stage=None):
        ...
