# tests/conftest.py

"""Shared fixtures.

The suite is offline by design: the datasets are 2-5 GB each, so the loaders are
exercised against tiny synthetic ``.h5ad`` files and every download is stubbed.
Two autouse guards enforce that, because ZENODO_RECORD_ID points at a real
published record and it is otherwise easy to turn a stubbed test into a multi-GB
transfer without noticing.
"""

# -- import packages: ---------------------------------------------------------
import anndata
import numpy as np
import pandas as pd
import pytest


# Large enough that PCA(n_components=50) is well-defined after gene filtering
# keeps half the vars, and still small enough to be instant.
N_OBS = 120
N_VARS = 120


@pytest.fixture(autouse=True)
def zenodo_disabled(request, monkeypatch):
    """Point the downloader at no Zenodo record for offline tests.

    Tests that exercise the Zenodo path patch a fake record id back on themselves.
    """
    if "slow" in request.keywords:
        return
    monkeypatch.setattr(
        "scdiffeq.datasets._figshare_downloader.ZENODO_RECORD_ID", None
    )


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Fail loudly instead of hanging if an offline test tries to use the network.

    Tests marked ``slow`` are exempt: they are opt-in and genuinely hit the
    network.
    """
    if "slow" in request.keywords:
        return

    import socket

    def _blocked(self, *args, **kwargs):
        raise RuntimeError(
            "network access attempted in an offline test; stub the request or "
            "mark the test @pytest.mark.slow"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


@pytest.fixture
def make_adata():
    """Factory for a small AnnData resembling the raw LARRY layout."""

    def _make(n_obs: int = N_OBS, n_vars: int = N_VARS, with_pca: int = 0, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_obs, n_vars)).astype(np.float32)

        obs = pd.DataFrame(index=[str(i) for i in range(n_obs)])
        var = pd.DataFrame(
            # Half the genes pass the filter, so filtering is observable.
            {"use_genes": [i % 2 == 0 for i in range(n_vars)]},
            index=[f"gene_{i}" for i in range(n_vars)],
        )

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        if with_pca:
            adata.obsm["X_pca"] = rng.normal(size=(n_obs, with_pca))
        return adata

    return _make


@pytest.fixture
def cytotrace_csvs(tmp_path):
    """Write the two CytoTRACE annotation CSVs the loader would otherwise download.

    Their presence on disk means the loader's ``.exists()`` guard short-circuits
    the download, which keeps these tests offline.
    """

    def _write(data_dir, n_obs: int = N_OBS, n_vars: int = N_VARS):
        data_dir.mkdir(parents=True, exist_ok=True)

        obs_df = pd.DataFrame(
            {"ct_pseudotime": np.linspace(0, 1, n_obs)},
            index=[str(i) for i in range(n_obs)],
        )
        obs_df.to_csv(data_dir / "larry.ct_obs_df.csv")

        var_df = pd.DataFrame(
            {"ct_gene_corr": np.linspace(-1, 1, n_vars)},
            index=[f"gene_{i}" for i in range(n_vars)],
        )
        var_df.to_csv(data_dir / "larry.ct_var_df.csv")

    return _write
