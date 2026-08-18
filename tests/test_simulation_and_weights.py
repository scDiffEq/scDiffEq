# tests/test_simulation_and_weights.py

"""Guards for three ways a fate-prediction run can go wrong without saying so.

Each of these surfaced while reproducing the published LARRY fate-prediction
benchmark: a missing weight column silently became no weighting, a t0-only
``adata`` silently integrated nothing, and two trajectories could silently share
a ``sim`` id.
"""

# -- import packages: ---------------------------------------------------------
import anndata
import logging
import numpy as np
import pandas as pd
import pytest


# -- fixtures: ----------------------------------------------------------------
@pytest.fixture
def time_course():
    """A LARRY-shaped object: three timepoints, a 50-dim basis."""
    rng = np.random.default_rng(0)
    n, d = 90, 50
    X = rng.normal(size=(n, d)).astype(np.float32)
    obs = pd.DataFrame(
        {"Time point": np.repeat([2.0, 4.0, 6.0], n // 3)},
        index=[str(i) for i in range(n)],
    )
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X
    return adata


# -- tests: -------------------------------------------------------------------
def test_missing_weight_key_warns(time_course, caplog):
    """A weight_key that names no obs column means every cell weighted equally.

    That is a legitimate configuration -- it is the unweighted arm -- but it has
    to be said out loud, because it is also what a typo produces.
    """
    import scdiffeq as sdq

    with caplog.at_level(logging.WARNING):
        model = sdq.scDiffEq(
            adata=time_course,
            latent_dim=50,
            use_key="X_pca",
            time_key="Time point",
            weight_key="KEGG",
            mu_hidden=[8, 8],
            sigma_hidden=[4, 4],
            silent=True,
        )

    assert "KEGG" in caplog.text
    assert np.unique(model.adata.obs["KEGG"].values).tolist() == [1]


def test_obs_keys_default_is_not_mutated(time_course):
    """Building one model must not change what the next one loads.

    ``obs_keys`` defaults to a single shared list object; appending the weight
    key to it in place made every later model in the session carry the previous
    model's weight key, and fail on it.
    """
    import inspect

    import scdiffeq as sdq

    default = inspect.signature(sdq.scDiffEq.__init__).parameters["obs_keys"].default

    sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        weight_key="KEGG",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )

    assert default == []
    # a second model, with no weight_key of its own, must still start clean
    second = sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )
    assert second.LitDataModule._obs_keys == ["W"]


def test_present_weight_key_is_untouched(time_course):
    """Supplied weights survive configuration."""
    import scdiffeq as sdq

    rng = np.random.default_rng(1)
    W = rng.uniform(0.5, 1.5, size=len(time_course))
    time_course.obs["KEGG"] = W

    model = sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        weight_key="KEGG",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )

    np.testing.assert_allclose(model.adata.obs["KEGG"].values, W)


def test_simulate_rejects_a_collapsed_time_span(time_course):
    """``idx`` picks the starting cells; ``adata`` supplies the span.

    Subsetting ``adata`` to the t0 cells -- the intuitive way to say "start
    here" -- used to return the input unintegrated.
    """
    import scdiffeq as sdq

    model = sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )
    t0_idx = time_course.obs.index[time_course.obs["Time point"] == 2.0]
    t0_only = time_course[t0_idx].copy()

    with pytest.raises(ValueError, match="single value"):
        sdq.tl.simulate(
            adata=t0_only,
            diffeq=model.DiffEq,
            idx=t0_idx,
            use_key="X_pca",
            time_key="Time point",
            N=2,
        )


def test_simulate_spans_the_full_time_course(time_course):
    """The default grid runs t_min -> t_max of the passed object at dt."""
    import scdiffeq as sdq

    model = sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )
    t0_idx = time_course.obs.index[time_course.obs["Time point"] == 2.0][:3]
    N = 4

    sim = sdq.tl.simulate(
        adata=time_course,
        diffeq=model.DiffEq,
        idx=t0_idx,
        use_key="X_pca",
        time_key="Time point",
        N=N,
    )

    t = np.unique(sim.obs["t"])
    assert (t.min(), t.max()) == (2.0, 6.0)
    assert len(t) == 41  # (6 - 2) / 0.1 + 1
    # the terminal state is one row per (progenitor, trajectory)
    assert (sim.obs["t"] == t.max()).sum() == len(t0_idx) * N


def test_sim_id_survives_indices_of_unequal_length(time_course):
    """'13199' + '90' and '131999' + '0' must not name the same trajectory."""
    import scdiffeq as sdq

    idx = time_course.obs.index.tolist()
    idx[0], idx[1] = "13199", "131999"
    time_course.obs.index = pd.Index(idx)

    model = sdq.scDiffEq(
        adata=time_course,
        latent_dim=50,
        use_key="X_pca",
        time_key="Time point",
        mu_hidden=[8, 8],
        sigma_hidden=[4, 4],
        silent=True,
    )

    sim = sdq.tl.simulate(
        adata=time_course,
        diffeq=model.DiffEq,
        idx=pd.Index(["13199", "131999"]),
        use_key="X_pca",
        time_key="Time point",
        N=100,
    )

    pairs = sim.obs[["z0_idx", "sim_i"]].apply(tuple, axis=1)
    assert sim.obs["sim"].nunique() == pairs.nunique()
