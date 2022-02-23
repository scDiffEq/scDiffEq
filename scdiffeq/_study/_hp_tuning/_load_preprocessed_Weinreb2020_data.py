import anndata as a
import licorice
import pickle


class _Weinreb2020:
    def __init__(self, path):

        self._adata_path = path
        self._stripped_path = ".".join(path.split(".")[:-1])
        self._umap_path = self._stripped_path + ".umap"
        self._pca_path = self._stripped_path + ".pca"

    def load(self):

        self.adata = a.read_h5ad(self._adata_path)
        self.umap = pickle.load(open(self._umap_path, "rb"))
        self.pca = pickle.load(open(self._pca_path, "rb"))

        self._data = [self.adata, self.umap, self.pca]
        self._names = ["AnnData", "UMAP", "PCA"]
        for i in range(len(self._data)):
            print(
                "\n{}:\n{}".format(
                    licorice.font_format(self._names[i], ["BOLD"]), self._data[i]
                )
            )


def _load_preprocessed_Weinreb2020_data(path):

    """"""

    data = _Weinreb2020(path)
    data.load()

    return data