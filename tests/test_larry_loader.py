# tests/test_larry_loader.py

"""Regression tests for the LARRY loader's caching and preprocessing.

Two defects are covered here, both reported against
``scdiffeq.datasets.larry(variant="fate_prediction")``:

1. Preprocessing ran only on the download path, so an ``.h5ad`` already on disk --
   exactly what you get when working around a broken downloader by fetching the
   file by hand -- was returned raw, with no ``X_pca``.
2. The raw download and the preprocessed result shared one path, so a preprocess
   that failed partway left a raw file at the processed path, and every later call
   silently returned it.

All tests are offline: the raw file is planted on disk and the CytoTRACE CSVs are
pre-written, so no download is attempted.
"""

# -- import packages: ---------------------------------------------------------
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

# -- import local dependencies: -----------------------------------------------
from scdiffeq.datasets import larry
from scdiffeq.datasets._larry_in_vitro import (
    LARRYInVitroDataset,
    _merge_annotations,
    _probe_h5ad,
    _resolve_variant,
)


@pytest.fixture
def planted(tmp_path, make_adata, cytotrace_csvs):
    """Plant a raw .h5ad at the loader's raw path and stage the CytoTRACE CSVs."""

    def _plant(variant=None, **kwargs):
        handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant=variant, **kwargs)
        cytotrace_csvs(handler.data_dir)
        make_adata().write_h5ad(handler.raw_h5ad_path)
        return handler

    return _plant


# -- Defect 1: preprocessing was skipped for files already on disk ------------
def test_preexisting_raw_file_is_preprocessed(planted, tmp_path):
    """The reported bug: a cached raw file must still come back with X_pca."""

    planted()
    adata = larry(data_dir=str(tmp_path))

    assert "X_pca" in adata.obsm, "a pre-existing raw file must still be preprocessed"
    assert adata.obsm["X_pca"].shape[1] == LARRYInVitroDataset.N_PCS
    assert "X_scaled" in adata.obsm, "perturbation tools default to use_key='X_scaled'"


def test_raw_and_processed_are_separate_files(planted):
    """Preprocessing must not overwrite the raw download."""

    handler = planted()
    _ = handler.adata

    assert handler.raw_h5ad_path.exists()
    assert handler.processed_h5ad_path.exists()
    assert handler.raw_h5ad_path != handler.processed_h5ad_path

    # The raw file must remain raw, so the processed file can be regenerated.
    assert _probe_h5ad(handler.raw_h5ad_path).pca_dim is None


def test_gene_filtering_is_applied(planted):
    handler = planted()
    adata = handler.adata

    raw_n_vars = _probe_h5ad(handler.raw_h5ad_path).n_vars
    assert adata.n_vars < raw_n_vars


def test_cytotrace_runs_without_the_other_flags(planted):
    """cytotrace=True used to be silently skipped unless another flag was set."""

    handler = planted(filter_genes=False, reduce_dimensions=False, cytotrace=True)
    adata = handler.adata

    assert "ct_pseudotime" in adata.obs.columns
    assert adata.obs["ct_pseudotime"].dtype.kind == "f", "must not be cast to str"


# -- Defect 2: a bad processed file was returned forever ----------------------
def test_invalid_processed_file_is_regenerated(planted):
    """A raw (or truncated) file sitting at the processed path must self-heal."""

    handler = planted()
    # Simulate a preprocess that died partway and left raw bytes at the target.
    handler.processed_h5ad_path.write_bytes(handler.raw_h5ad_path.read_bytes())

    is_valid, reason = handler._validate_processed(handler.processed_h5ad_path)
    assert not is_valid
    assert "X_pca" in reason

    adata = handler.adata
    assert "X_pca" in adata.obsm


def test_truncated_processed_file_is_regenerated(planted):
    handler = planted()
    handler.processed_h5ad_path.write_bytes(b"not an hdf5 file")

    is_valid, reason = handler._validate_processed(handler.processed_h5ad_path)
    assert not is_valid
    assert "unreadable" in reason

    assert "X_pca" in handler.adata.obsm


def test_valid_processed_cache_is_reused(planted, monkeypatch):
    """A good cache must not trigger another preprocess."""

    handler = planted()
    _ = handler.adata

    second = LARRYInVitroDataset(data_dir=handler._data_dir)

    def _boom(*args, **kwargs):
        raise AssertionError("preprocessing re-ran despite a valid cache")

    monkeypatch.setattr(second, "_preprocess", _boom)
    assert "X_pca" in second.adata.obsm


def test_non_default_flags_get_their_own_cache(planted):
    """reduce_dimensions=False must not poison the shared processed path."""

    default = planted()
    nopca = LARRYInVitroDataset(data_dir=default._data_dir, reduce_dimensions=False)

    assert default.processed_h5ad_path != nopca.processed_h5ad_path
    assert "X_pca" not in nopca.adata.obsm
    assert "X_pca" in default.adata.obsm


def test_reduce_dimensions_false_does_not_loop(planted):
    """Validity must not demand an X_pca that was never requested."""

    handler = planted(reduce_dimensions=False)
    _ = handler.adata

    is_valid, reason = handler._validate_processed(handler.processed_h5ad_path)
    assert is_valid, reason


# -- property semantics -------------------------------------------------------
def test_adata_property_is_stable_across_accesses(planted):
    """The property used to return None on its second access."""

    handler = planted()
    first = handler.adata
    second = handler.adata

    assert second is not None
    assert second is first


# -- determinism (Fix 2) ------------------------------------------------------
def test_pca_is_deterministic(tmp_path, make_adata, cytotrace_csvs):
    """Two independent preprocessing runs must produce an identical basis."""

    results = []
    for i in range(2):
        data_dir = tmp_path / f"run{i}"
        handler = LARRYInVitroDataset(data_dir=str(data_dir))
        cytotrace_csvs(handler.data_dir)
        make_adata().write_h5ad(handler.raw_h5ad_path)
        results.append(handler.adata.obsm["X_pca"])

    np.testing.assert_array_equal(results[0], results[1])


# -- legacy cache migration (Fix 1, migration path) ---------------------------
def test_legacy_raw_cache_is_migrated_not_redownloaded(tmp_path, make_adata, cytotrace_csvs):
    """A pre-existing larry.h5ad without X_pca is adopted as the raw input."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path))
    cytotrace_csvs(handler.data_dir)
    legacy = handler.data_dir / "larry.h5ad"
    make_adata().write_h5ad(legacy)

    adata = handler.adata

    assert not legacy.exists(), "legacy file should have been renamed"
    assert handler.raw_h5ad_path.exists()
    assert "X_pca" in adata.obsm


def test_legacy_processed_cache_is_migrated(tmp_path, make_adata, cytotrace_csvs):
    """A pre-existing file that already has a width-50 X_pca is treated as processed."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path))
    cytotrace_csvs(handler.data_dir)
    legacy = handler.data_dir / "larry.h5ad"

    adata = make_adata(with_pca=LARRYInVitroDataset.N_PCS)
    adata.obs["ct_pseudotime"] = np.linspace(0, 1, adata.n_obs)
    adata.write_h5ad(legacy)

    handler._migrate_legacy_cache()

    assert not legacy.exists()
    assert handler.processed_h5ad_path.exists()
    assert not handler.raw_h5ad_path.exists(), "must not re-download multi-GB raw data"


def test_legacy_file_with_wrong_pca_width_is_treated_as_raw(tmp_path, make_adata, cytotrace_csvs):
    """Presence of X_pca alone is not enough - the width is the discriminator."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path))
    cytotrace_csvs(handler.data_dir)
    legacy = handler.data_dir / "larry.h5ad"
    make_adata(with_pca=2).write_h5ad(legacy)

    handler._migrate_legacy_cache()

    assert handler.raw_h5ad_path.exists()
    assert not handler.processed_h5ad_path.exists()


def test_unreadable_legacy_file_is_left_alone(tmp_path, make_adata, cytotrace_csvs):
    handler = LARRYInVitroDataset(data_dir=str(tmp_path))
    cytotrace_csvs(handler.data_dir)
    legacy = handler.data_dir / "larry.h5ad"
    legacy.write_bytes(b"garbage")

    handler._migrate_legacy_cache()

    assert legacy.exists(), "a file we cannot classify must not be moved"
    assert not handler.raw_h5ad_path.exists()


# -- variant naming (Fix 4) ---------------------------------------------------
def test_fate_prediction_alias_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="unprocessed"):
        assert _resolve_variant("fate_prediction") == "unprocessed"


def test_canonical_variants_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _resolve_variant(None) is None
        assert _resolve_variant("unprocessed") == "unprocessed"


def test_unknown_variant_raises_a_useful_error():
    with pytest.raises(ValueError, match="Unknown variant"):
        _resolve_variant("nope")


def test_variant_paths_do_not_collide(tmp_path):
    default = LARRYInVitroDataset(data_dir=str(tmp_path), variant=None)
    filtered = LARRYInVitroDataset(data_dir=str(tmp_path), variant="unprocessed")

    assert default.raw_h5ad_path != filtered.raw_h5ad_path
    assert default.processed_h5ad_path != filtered.processed_h5ad_path
    # Fitted models used to clobber each other across variants.
    assert default._model_path("pca") != filtered._model_path("pca")


def test_default_variant_keeps_documented_pickle_names(tmp_path):
    """quickstart.ipynb loads scaler.pkl / pca.pkl by these exact names."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path))
    assert handler._model_path("scaler").name == "scaler.pkl"
    assert handler._model_path("pca").name == "pca.pkl"


def test_deprecated_alias_reaches_the_unprocessed_paths(tmp_path):
    with pytest.warns(DeprecationWarning):
        handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant="fate_prediction")

    assert handler._variant == "unprocessed"
    assert "unprocessed" in handler.raw_h5ad_path.name


# -- gene-filtered variant without use_genes ----------------------------------
def test_missing_use_genes_is_a_no_op_not_a_crash(tmp_path, make_adata, cytotrace_csvs):
    """The gene-filtered variant ships no use_genes column; that must not raise."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant="unprocessed")
    cytotrace_csvs(handler.data_dir)

    adata = make_adata()
    del adata.var["use_genes"]
    adata.write_h5ad(handler.raw_h5ad_path)

    result = handler.adata
    assert "X_pca" in result.obsm
    assert result.n_vars == adata.n_vars


# -- annotation merge semantics ------------------------------------------------
def test_partial_annotation_coverage_preserves_frame_length():
    """The real var annotations cover only the filtered gene subset."""

    target = pd.DataFrame(index=[str(i) for i in range(10)])
    annotations = pd.DataFrame(
        {"ct_gene_corr": [0.1, 0.2, 0.3]}, index=["1", "3", "5"]
    )

    merged = _merge_annotations(target, annotations, axis_name="var")

    assert len(merged) == 10, "pd.concat used to take the union and change length"
    assert merged.loc["1", "ct_gene_corr"] == pytest.approx(0.1)
    assert pd.isna(merged.loc["0", "ct_gene_corr"])


def test_annotations_outside_the_target_are_dropped():
    """Annotation keys absent from the target must not lengthen the frame."""

    target = pd.DataFrame(index=["0", "1"])
    annotations = pd.DataFrame(
        {"ct_pseudotime": [0.1, 0.2, 0.3]}, index=["0", "1", "999"]
    )

    merged = _merge_annotations(target, annotations, axis_name="obs")

    assert list(merged.index) == ["0", "1"]


def test_partially_covered_bool_column_stays_writable(tmp_path, make_adata):
    """A bool annotation with gaps must not become an unwritable object column.

    The real ct_correlates column is bool and covers only 2,492 of the unfiltered
    variant's 25,289 genes. Reindexing upcasts bool -> object with NaN, which h5py
    rejects with "Can't implicitly convert non-string objects to strings".
    """

    adata = make_adata()
    annotations = pd.DataFrame(
        {"ct_correlates": [True, False]}, index=list(adata.var_names[:2])
    )

    adata.var = _merge_annotations(adata.var, annotations, axis_name="var")

    # The real assertion: this must not raise.
    out = tmp_path / "written.h5ad"
    adata.write_h5ad(out)
    assert out.exists()


def test_partially_covered_numeric_column_keeps_nan(make_adata):
    adata = make_adata()
    annotations = pd.DataFrame(
        {"ct_gene_corr": [0.5, -0.5]}, index=list(adata.var_names[:2])
    )

    merged = _merge_annotations(adata.var, annotations, axis_name="var")

    assert merged["ct_gene_corr"].dtype.kind == "f"
    assert merged["ct_gene_corr"].iloc[0] == pytest.approx(0.5)
    assert pd.isna(merged["ct_gene_corr"].iloc[-1])


def test_repeated_merge_does_not_duplicate_columns():
    """Re-running used to produce duplicate names, breaking .astype(float)."""

    target = pd.DataFrame(index=["0", "1"])
    annotations = pd.DataFrame({"ct_pseudotime": [0.1, 0.2]}, index=["0", "1"])

    merged = _merge_annotations(target, annotations, axis_name="obs")
    merged = _merge_annotations(merged, annotations, axis_name="obs")

    assert list(merged.columns) == ["ct_pseudotime"]
    assert isinstance(merged["ct_pseudotime"], pd.Series)


# -- prebuilt processed artifact ----------------------------------------------
def test_no_variant_registers_a_prebuilt_artifact(tmp_path, make_adata, cytotrace_csvs):
    """Currently none should: the prebuilt file costs more to fetch than to build.

    2.93 GB prebuilt vs 2.16 GB raw + ~30 s of local preprocessing, at ~2.7 MB/s.
    """

    for variant in (None, "unprocessed"):
        handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant=variant)
        assert handler._spec.get("processed_fname") is None
        assert handler._try_download_processed() is False


def test_prebuilt_artifact_is_used_when_registered(
    tmp_path, make_adata, cytotrace_csvs, monkeypatch
):
    """The mechanism still works, so re-enabling it is a one-line change."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant="unprocessed")
    cytotrace_csvs(handler.data_dir)
    monkeypatch.setitem(
        handler._spec, "processed_fname", "larry_unprocessed.processed.h5ad"
    )

    prebuilt = make_adata(with_pca=LARRYInVitroDataset.N_PCS)
    prebuilt.obs["ct_pseudotime"] = np.linspace(0, 1, prebuilt.n_obs)

    def _fake_download(filename, write_path, **kwargs):
        prebuilt.write_h5ad(write_path)
        return True

    monkeypatch.setattr(
        "scdiffeq.datasets._larry_in_vitro.zenodo_file_downloader", _fake_download
    )

    assert handler._try_download_processed() is True
    assert handler.processed_h5ad_path.exists()


def test_prebuilt_artifact_not_used_for_non_default_flags(tmp_path):
    """The prebuilt object only matches the defaults it was built with."""

    handler = LARRYInVitroDataset(
        data_dir=str(tmp_path), variant="unprocessed", reduce_dimensions=False
    )
    assert handler._pp_tag  # non-default
    assert handler._try_download_processed() is False


def test_invalid_prebuilt_artifact_falls_back_to_local(
    tmp_path, make_adata, cytotrace_csvs, monkeypatch
):
    """A corrupt prebuilt download must not be returned or left on disk."""

    handler = LARRYInVitroDataset(data_dir=str(tmp_path), variant="unprocessed")
    cytotrace_csvs(handler.data_dir)
    make_adata().write_h5ad(handler.raw_h5ad_path)

    def _fake_download(filename, write_path, **kwargs):
        pathlib.Path(write_path).write_bytes(b"not an hdf5 file")
        return True

    monkeypatch.setattr(
        "scdiffeq.datasets._larry_in_vitro.zenodo_file_downloader", _fake_download
    )

    assert handler._try_download_processed() is False
    assert not handler.processed_h5ad_path.exists(), "bad artifact must be removed"
    assert "X_pca" in handler.adata.obsm


# -- opt-in integration -------------------------------------------------------
@pytest.mark.slow
def test_larry_unprocessed_end_to_end(tmp_path):
    """Downloads ~2.2 GB. Run with `pytest -m slow`."""

    adata = larry(variant="unprocessed", data_dir=str(tmp_path))

    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape[1] == LARRYInVitroDataset.N_PCS
