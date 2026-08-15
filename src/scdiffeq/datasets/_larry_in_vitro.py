# -- import packages: ----------------------------------------------------------
import ABCParse
import anndata
import dataclasses
import h5py
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import sklearn.decomposition
import sklearn.preprocessing
import warnings


# -- import local dependencies: ------------------------------------------------
from .. import io
from ._figshare_downloader import figshare_downloader, zenodo_file_downloader

# -- set type hints: -----------------------------------------------------------
from typing import Dict, Optional, Tuple, Union

# -- configure logger: ----------------------------------------------------------
logger = logging.getLogger(__name__)


# -- variant registry: ---------------------------------------------------------
_VARIANTS = {
    None: {
        "figshare_id": 55415231,
        "stem": "larry",
        "legacy_fnames": ("larry.h5ad",),
    },
    "unprocessed": {
        "figshare_id": 52612805,
        "stem": "larry_unprocessed",
        "legacy_fnames": ("larry_fate_prediction.h5ad",),
        # Prebuilt equivalent of running the default preprocessing on this
        # variant; downloaded instead of recomputed when available.
        "processed_fname": "larry_unprocessed.processed.h5ad",
    },
}

# ``fate_prediction`` was a misleading name. It resolves to
# ``adata.Weinreb2020.in_vitro.gene_filtered.h5ad``, but despite that filename the
# object is 130,887 x 25,289 with no ``X_pca`` and a ``use_genes`` column to filter
# *by* -- it is the unprocessed input. The default variant is the one that is
# already filtered (2,492 genes) and ships a precomputed ``X_pca``.
_VARIANT_ALIASES = {"fate_prediction": "unprocessed"}


def _resolve_variant(variant: Optional[str]) -> Optional[str]:
    """Canonicalize ``variant``, warning once on a deprecated alias.

    Idempotent: re-resolving an already-canonical name never re-warns, which lets
    both ``larry()`` and ``LARRYInVitroDataset.__init__`` call it without emitting
    the warning twice.
    """
    if variant in _VARIANT_ALIASES:
        canonical = _VARIANT_ALIASES[variant]
        message = (
            f"variant={variant!r} is deprecated and will be removed in a future "
            f"release; use variant={canonical!r} instead. Note that this variant is "
            f"the unprocessed input (25,289 genes, no precomputed X_pca), not the "
            f"gene-filtered object its upstream filename suggests."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        # DeprecationWarning is hidden by default outside __main__, so also log it.
        logger.warning(message)
        return canonical

    if variant not in _VARIANTS:
        valid = sorted(
            repr(key) for key in list(_VARIANTS) + list(_VARIANT_ALIASES)
        )
        raise ValueError(
            f"Unknown variant: {variant!r}. Valid options: {', '.join(valid)}."
        )
    return variant


# -- lightweight, read-only h5ad introspection: --------------------------------
@dataclasses.dataclass(frozen=True)
class _H5ADProbe:
    """Metadata-only summary of an ``.h5ad`` file.

    Reads HDF5 metadata for a handful of nodes -- a few KB regardless of whether
    the file is 80 MB or 5.3 GB. This is what makes it viable to ask "is this
    cached file raw or already processed?" without loading it.
    """

    path: pathlib.Path
    readable: bool = False
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    obsm_keys: Tuple[str, ...] = ()
    obs_columns: Tuple[str, ...] = ()
    pca_dim: Optional[int] = None
    error: Optional[str] = None


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _frame_shape_and_columns(node) -> Tuple[Optional[int], Tuple[str, ...]]:
    """Length and column names of an h5ad-encoded DataFrame, reading no column data."""
    if isinstance(node, h5py.Dataset):  # legacy structured-array layout
        return int(node.shape[0]), tuple(node.dtype.names or ())

    index_key = _decode(node.attrs.get("_index", "_index"))

    n = None
    if index_key in node:
        index_node = node[index_key]
        if isinstance(index_node, h5py.Dataset):
            n = int(index_node.shape[0])
        elif "codes" in index_node:  # categorical index
            n = int(index_node["codes"].shape[0])

    columns = node.attrs.get("column-order")
    if columns is None:
        columns = [key for key in node.keys() if key != index_key]

    return n, tuple(_decode(col) for col in columns)


def _probe_h5ad(path: Union[str, pathlib.Path]) -> _H5ADProbe:
    """Open an ``.h5ad`` read-only and report its structure."""
    path = pathlib.Path(path)

    if (not path.exists()) or path.stat().st_size == 0:
        return _H5ADProbe(path=path, error="missing or empty")

    try:
        with h5py.File(path, "r") as f:
            n_obs, obs_columns = _frame_shape_and_columns(f["obs"])
            n_vars, _ = _frame_shape_and_columns(f["var"])

            obsm = f.get("obsm")
            obsm_keys = tuple(obsm.keys()) if obsm is not None else ()

            pca_dim = None
            if obsm is not None and "X_pca" in obsm:
                pca_node = obsm["X_pca"]
                if isinstance(pca_node, h5py.Dataset) and pca_node.ndim == 2:
                    pca_dim = int(pca_node.shape[1])

            return _H5ADProbe(
                path=path,
                readable=True,
                n_obs=n_obs,
                n_vars=n_vars,
                obsm_keys=obsm_keys,
                obs_columns=obs_columns,
                pca_dim=pca_dim,
            )
    except (OSError, KeyError, ValueError) as e:
        # truncated download, or a non-HDF5 payload such as a WAF challenge page
        return _H5ADProbe(path=path, error=f"{type(e).__name__}: {e}")


# -- atomic write helpers: -----------------------------------------------------
def _tmp_sibling(path: pathlib.Path, suffix: str) -> pathlib.Path:
    """Hidden sibling path in the same directory, so ``os.replace`` stays atomic."""
    return path.with_name(f".{path.name}.{suffix}")


def _write_h5ad_atomic(adata: anndata.AnnData, write_path: pathlib.Path) -> None:
    """Write to a temp sibling, then move into place.

    Preprocessing can fail partway -- CytoTRACE annotation pulls two CSVs over the
    network -- and without this a half-written file would be left at the processed
    path and silently returned by every later call.
    """
    tmp = _tmp_sibling(write_path, "tmp.h5ad")
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, write_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# -- CytoTRACE annotation: -----------------------------------------------------
_CYTOTRACE_FILES = {
    "obs": (54312011, "larry.ct_obs_df.csv"),
    "var": (54312008, "larry.ct_var_df.csv"),
}

# Columns the annotation contributes to ``adata.obs`` (used to validate a cache).
CYTOTRACE_OBS_COLS = ("ct_pseudotime",)


def _coerce_annotation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """String index, but leave numeric columns numeric.

    The previous implementation cast *every* column to ``str``, which is why
    ``ct_pseudotime`` had to be cast back to float on read.
    """
    df = df.copy()
    df.index = df.index.astype(str)
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric if numeric.notna().all() else df[col].astype(str)
    return df


def _download_cytotrace_annotations(
    data_dir: Union[str, pathlib.Path],
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch (once) and load the precomputed CytoTRACE obs/var annotations."""
    data_dir = pathlib.Path(data_dir)

    frames = {}
    for key, (figshare_id, fname) in _CYTOTRACE_FILES.items():
        path = data_dir.joinpath(fname)
        if force or (not path.exists()) or path.stat().st_size == 0:
            logger.info(f"Downloading CytoTRACE {key} annotations -> {path}")
            figshare_downloader(figshare_id=figshare_id, write_path=path)
        else:
            logger.debug(f"Using cached CytoTRACE {key} annotations: {path}")
        frames[key] = _coerce_annotation_frame(pd.read_csv(path, index_col=0))

    return frames


def _merge_annotations(
    target: pd.DataFrame,
    annotations: pd.DataFrame,
    axis_name: str,
) -> pd.DataFrame:
    """Index-safe left-join of annotations onto ``obs``/``var``.

    Replaces ``pd.concat(..., axis=1)``, which took the *union* of the two indexes:
    any annotation key absent from the target silently lengthened the frame (an
    AnnData assignment error), and re-running duplicated column names, after which
    ``adata.obs['ct_pseudotime']`` returns a DataFrame rather than a Series.

    Reindexing to the target keeps its length fixed and leaves NaN wherever an
    annotation is unavailable. The var annotations in particular cover only the
    gene-filtered subset, so partial coverage is normal and not an error.
    """
    target = target.copy()
    target.index = target.index.astype(str)

    overlap = int(target.index.isin(annotations.index).sum())
    if overlap == 0:
        logger.warning(
            f"No CytoTRACE {axis_name} annotations matched: none of {len(target)} "
            f"{axis_name} names are present in the annotation table. The columns "
            f"will be added but left empty."
        )
    elif overlap < len(target):
        logger.info(
            f"CytoTRACE {axis_name} annotations cover {overlap} of {len(target)} "
            f"{axis_name}; the remainder will be NaN."
        )

    # Left join: never changes len(target), never duplicates columns.
    aligned = annotations.reindex(target.index)
    for col in aligned.columns:
        if col in target.columns:
            logger.debug(f"Overwriting existing {axis_name} column {col!r}.")
        target[col] = _h5ad_safe_column(aligned[col]).values

    return target


def _h5ad_safe_column(values: pd.Series):
    """Coerce a reindexed column into something anndata can serialize.

    Partial coverage introduces NaN wherever an annotation was unavailable. A
    bool column then upcasts to ``object`` holding True/False/NaN, which h5py
    rejects with "Can't implicitly convert non-string objects to strings" -- the
    var annotations cover only the gene-filtered subset, so this is the normal
    case for the unfiltered variant, not an edge case.

    Numeric columns represent gaps as NaN natively and are left alone. Anything
    else becomes a categorical of strings, which encodes missingness properly.
    """
    if pd.api.types.is_bool_dtype(values):
        return values  # a real bool dtype cannot contain NaN

    if pd.api.types.is_numeric_dtype(values):
        return values

    present = values.notna()
    if present.all():
        # Fully covered object column: plain strings write cleanly.
        return values.astype(str)

    as_str = pd.Series(np.nan, index=values.index, dtype=object)
    as_str[present] = values[present].astype(str)
    return pd.Series(pd.Categorical(as_str), index=values.index)


def _annotate_larry_cytotrace(
    adata: anndata.AnnData,
    data_dir: Union[str, pathlib.Path],
    force: bool = False,
) -> anndata.AnnData:
    """Annotate ``adata`` with precomputed CytoTRACE obs/var values."""
    frames = _download_cytotrace_annotations(data_dir, force=force)
    adata.obs = _merge_annotations(adata.obs, frames["obs"], axis_name="obs")
    adata.var = _merge_annotations(adata.var, frames["var"], axis_name="var")
    return adata


# -- Controller class: ---------------------------------------------------------
class LARRYInVitroDataset(ABCParse.ABCParse):
    N_PCS = 50
    PCA_RANDOM_STATE = 0

    VARIANTS = _VARIANTS
    VARIANT_ALIASES = _VARIANT_ALIASES

    # Retained for backwards compatibility with any external caller.
    FIGSHARE_IDS = {
        None: 55415231,
        "gene_filtered": 52612805,
        "fate_prediction": 52612805,
    }

    def __init__(
        self,
        data_dir=os.getcwd(),
        variant: Optional[str] = None,
        filter_genes: bool = True,
        reduce_dimensions: bool = True,
        cytotrace: bool = True,
        force_download: bool = False,
        force_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        # Normalize before __parse__ so self._variant is always canonical.
        variant = _resolve_variant(variant)
        self.__parse__(locals())

    # -- directories: ---------------------------------------------------------
    @property
    def _scdiffeq_parent_data_dir(self) -> pathlib.Path:
        path = pathlib.Path(self._data_dir).joinpath("scdiffeq_data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> pathlib.Path:
        path = self._scdiffeq_parent_data_dir.joinpath("larry")
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- variant spec: --------------------------------------------------------
    @property
    def _spec(self) -> Dict:
        return self.VARIANTS[self._variant]

    @property
    def _stem(self) -> str:
        return self._spec["stem"]

    @property
    def _figshare_id(self) -> int:
        return self._spec["figshare_id"]

    @property
    def _DO_PREPROCESSING(self) -> bool:
        return any([self._filter_genes, self._reduce_dimensions, self._cytotrace])

    @property
    def _pp_tag(self) -> str:
        """Filename tag for non-default flag combinations ('' for the default).

        Without this, ``larry(reduce_dimensions=False)`` would write a PCA-less file
        to the shared processed path and poison the cache for every other caller.
        """
        tags = []
        if not self._filter_genes:
            tags.append("nofilter")
        if not self._reduce_dimensions:
            tags.append("nopca")
        if not self._cytotrace:
            tags.append("noct")
        return f".{'-'.join(tags)}" if tags else ""

    # -- paths: ---------------------------------------------------------------
    @property
    def raw_h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(f"_{self._stem}.raw.h5ad")

    @property
    def processed_h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(f"{self._stem}.processed{self._pp_tag}.h5ad")

    @property
    def h5ad_path(self) -> pathlib.Path:
        """Path the returned object is read from."""
        if self._DO_PREPROCESSING:
            return self.processed_h5ad_path
        return self.raw_h5ad_path

    def _model_path(self, kind: str) -> pathlib.Path:
        """Path for a fitted scaler/PCA.

        The default variant keeps the bare ``scaler.pkl`` / ``pca.pkl`` names that
        the published quickstart notebook loads. Other variants are namespaced so
        they no longer silently overwrite each other's models.
        """
        prefix = "" if self._variant is None else f"{self._stem}."
        return self.data_dir.joinpath(f"{prefix}{kind}{self._pp_tag}.pkl")

    # -- legacy cache migration: ----------------------------------------------
    def _migrate_legacy_cache(self) -> None:
        """Adopt a pre-existing ``larry.h5ad`` rather than re-downloading GBs.

        Existing installs hold a single file that may be raw *or* processed
        depending on how it got there, so classify it from HDF5 metadata instead of
        assuming. The move is an ``os.replace`` within one directory: instant and
        atomic even at 5.3 GB.
        """
        if getattr(self, "_migrated", False):
            return
        self._migrated = True

        for legacy_name in self._spec["legacy_fnames"]:
            legacy_path = self.data_dir.joinpath(legacy_name)

            if not legacy_path.exists():
                continue
            if legacy_path in (self.raw_h5ad_path, self.processed_h5ad_path):
                continue

            probe = _probe_h5ad(legacy_path)
            if not probe.readable:
                logger.warning(
                    f"Ignoring unreadable legacy cache file {legacy_path} "
                    f"({probe.error}). Delete it to reclaim disk space."
                )
                continue

            # Width is the discriminator, not mere presence: a raw upload may also
            # carry an X_pca of some other width.
            looks_processed = probe.pca_dim == self.N_PCS

            if looks_processed and self._pp_tag:
                logger.warning(
                    f"Legacy cache {legacy_path.name} appears to be preprocessed with "
                    f"default settings, but non-default preprocessing was requested; "
                    f"it cannot be reused. Leaving it in place."
                )
                continue

            target = self.processed_h5ad_path if looks_processed else self.raw_h5ad_path
            kind = "processed" if looks_processed else "raw"

            if target.exists():
                logger.info(
                    f"{target.name} already exists; leaving legacy file "
                    f"{legacy_path.name} in place (safe to delete)."
                )
                continue

            logger.info(
                f"Migrating legacy cache: {legacy_path.name} -> {target.name} "
                f"(detected as {kind}: n_obs={probe.n_obs}, n_vars={probe.n_vars}, "
                f"X_pca width={probe.pca_dim})"
            )
            os.replace(legacy_path, target)

    # -- validation: ----------------------------------------------------------
    def _validate_processed(self, path: pathlib.Path) -> Tuple[bool, str]:
        """Structural check on a cached processed file. Returns ``(is_valid, reason)``."""
        probe = _probe_h5ad(path)
        if not probe.readable:
            return False, f"unreadable ({probe.error})"

        if self._reduce_dimensions:
            if probe.pca_dim is None:
                return False, "missing obsm['X_pca']"
            if probe.pca_dim != self.N_PCS:
                return False, (
                    f"obsm['X_pca'] has {probe.pca_dim} components "
                    f"(expected {self.N_PCS})"
                )

        if self._cytotrace:
            missing = set(CYTOTRACE_OBS_COLS) - set(probe.obs_columns)
            if missing:
                return False, f"missing CytoTRACE obs columns: {sorted(missing)}"

        # Cross-check against raw only when raw is present, so a valid cache is
        # never invalidated merely because the raw file is unavailable.
        raw_probe = _probe_h5ad(self.raw_h5ad_path)
        if raw_probe.readable and probe.n_obs != raw_probe.n_obs:
            return False, (
                f"n_obs mismatch (processed={probe.n_obs}, raw={raw_probe.n_obs})"
            )

        return True, "ok"

    # -- raw acquisition: -----------------------------------------------------
    def download(self) -> None:
        """Download the raw file for this variant if it is not already cached."""
        if self.raw_h5ad_path.exists() and not self._force_download:
            return

        logger.info(
            f"Downloading LARRY (variant={self._variant!r}, "
            f"figshare_id={self._figshare_id}) -> {self.raw_h5ad_path}"
        )
        figshare_downloader(
            figshare_id=self._figshare_id,
            write_path=self.raw_h5ad_path,
        )
        # One download per instance, so a second property access does not re-fetch.
        self._force_download = False

    @property
    def raw_adata(self) -> anndata.AnnData:
        if not hasattr(self, "_raw_adata"):
            self._migrate_legacy_cache()
            self.download()
            logger.info(f"Reading raw data from {self.raw_h5ad_path}")
            self._raw_adata = anndata.read_h5ad(self.raw_h5ad_path)
        return self._raw_adata

    # -- preprocessing steps: -------------------------------------------------
    def _gene_filtering(self, adata: anndata.AnnData) -> anndata.AnnData:
        if "use_genes" not in adata.var.columns:
            # The gene-filtered variant is already filtered and carries no
            # use_genes column. Warn rather than raise, so this stays a no-op
            # instead of breaking the call.
            logger.warning(
                f"var['use_genes'] not found in {self.raw_h5ad_path.name}; skipping "
                f"gene filtering (this variant is likely already gene-filtered)."
            )
            return adata

        use_genes = adata.var["use_genes"]
        if use_genes.dtype != bool:
            use_genes = (
                use_genes.astype(str).str.strip().str.lower().isin(["true", "1"])
            )

        logger.info(f"Filtering genes: {adata.n_vars} -> {int(use_genes.sum())}")
        return adata[:, use_genes.values].copy()

    def _dimension_reduction(self, adata: anndata.AnnData) -> None:
        """Scale and run PCA, in place on ``adata.obsm``."""
        scaler = sklearn.preprocessing.StandardScaler()
        pca = sklearn.decomposition.PCA(
            n_components=self.N_PCS,
            random_state=self.PCA_RANDOM_STATE,
        )

        n_bytes = adata.n_obs * adata.n_vars * 8
        if n_bytes > 8e9:
            logger.warning(
                f"Dense scaling of {adata.n_obs} x {adata.n_vars} needs roughly "
                f"{n_bytes / 1e9:.1f} GB of RAM, and the same again on disk in "
                f"obsm['X_scaled']. Consider filter_genes=True."
            )

        X_raw = adata.X
        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.toarray()

        adata.obsm["X_scaled"] = scaler.fit_transform(X_raw)
        adata.obsm["X_pca"] = pca.fit_transform(adata.obsm["X_scaled"])

        io.write_pickle(obj=scaler, path=self._model_path("scaler"))
        io.write_pickle(obj=pca, path=self._model_path("pca"))

    def _preprocess(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Run each requested step, then persist the result.

        Ordering is load-bearing: CytoTRACE first, so its var-level annotations
        survive gene filtering; PCA last, so the scaler and PCA are fit on the
        filtered gene space.
        """
        logger.info(f"Preprocessing LARRY (variant={self._variant!r})...")

        if self._cytotrace:
            adata = _annotate_larry_cytotrace(
                adata=adata,
                data_dir=self.data_dir,
                force=self._force_download,
            )
        if self._filter_genes:
            adata = self._gene_filtering(adata)
        if self._reduce_dimensions:
            self._dimension_reduction(adata)

        logger.info(f"Writing processed data to {self.processed_h5ad_path}")
        _write_h5ad_atomic(adata, self.processed_h5ad_path)

        return adata

    def _try_download_processed(self) -> bool:
        """Fetch a prebuilt processed artifact instead of recomputing it.

        Only valid when the requested flags match the defaults the artifact was
        built with -- otherwise the prebuilt object is not what was asked for and
        preprocessing has to run locally. Returns ``False`` whenever the artifact
        is unavailable, so this is always an optimization, never a requirement.
        """
        processed_fname = self._spec.get("processed_fname")
        if not processed_fname or self._pp_tag:
            return False

        logger.info(f"Checking for a prebuilt {processed_fname}...")
        if not zenodo_file_downloader(
            filename=processed_fname,
            write_path=self.processed_h5ad_path,
        ):
            return False

        is_valid, reason = self._validate_processed(self.processed_h5ad_path)
        if not is_valid:
            logger.warning(
                f"Prebuilt {processed_fname} failed validation ({reason}); "
                f"falling back to local preprocessing."
            )
            self.processed_h5ad_path.unlink(missing_ok=True)
            return False

        return True

    def _ensure_processed(self) -> None:
        """Guarantee that ``self.h5ad_path`` exists and is valid."""
        self._migrate_legacy_cache()

        if not self._DO_PREPROCESSING:
            self.download()
            return

        if self._force_download and self.processed_h5ad_path.exists():
            logger.info(
                f"force_download=True: discarding stale "
                f"{self.processed_h5ad_path.name}"
            )
            self.processed_h5ad_path.unlink()

        if self.processed_h5ad_path.exists() and not self._force_preprocess:
            is_valid, reason = self._validate_processed(self.processed_h5ad_path)
            if is_valid:
                return
            logger.warning(
                f"Cached {self.processed_h5ad_path.name} failed validation "
                f"({reason}); regenerating from raw."
            )
            self.processed_h5ad_path.unlink()

        if self._try_download_processed():
            return

        adata = self.raw_adata  # downloads only if the raw file is absent
        self._preprocess(adata=adata)

        # Release the large intermediates before re-reading from disk.
        del adata
        if hasattr(self, "_raw_adata"):
            del self._raw_adata

    def _safe_read(self, path: Optional[pathlib.Path] = None) -> anndata.AnnData:
        path = self.h5ad_path if path is None else path
        logger.info(f"Loading data from {path}")
        try:
            adata = anndata.read_h5ad(path)
        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            raise

        # Defensive: files written before the CytoTRACE dtype fix stored these as str.
        for col in CYTOTRACE_OBS_COLS:
            if col in adata.obs.columns:
                adata.obs[col] = pd.to_numeric(adata.obs[col], errors="coerce")

        adata.obs.index.name = "index"
        return adata

    @property
    def adata(self) -> anndata.AnnData:
        """Preprocessed LARRY in vitro AnnData for the requested variant."""
        if not hasattr(self, "_adata"):
            self._ensure_processed()
            self._adata = self._safe_read()
        return self._adata


def larry(
    data_dir: str = os.getcwd(),
    variant: Optional[str] = None,
    filter_genes: bool = True,
    reduce_dimensions: bool = True,
    cytotrace: bool = True,
    force_download: bool = False,
    force_preprocess: bool = False,
) -> anndata.AnnData:
    """LARRY in vitro dataset

    The raw download and the preprocessed result are cached as separate files, so
    a dataset obtained by any means (including a manual download) is still
    preprocessed on first use.

    Args:
        data_dir: str, default=os.getcwd()
            Path to the directory where the data will be saved.
        variant: Optional[str], default=None
            Dataset variant.

            ``None`` (default) is the biology-rich object: 130,887 x 2,492, already
            gene-filtered, shipping a precomputed ``X_pca``, ``X_umap`` and
            ``X_scaled``.

            ``"unprocessed"`` is the upstream input: 130,887 x 25,289 with no
            ``X_pca``, carrying a ``use_genes`` column that preprocessing filters
            by (down to 2,447 genes). ``X_pca`` is computed locally, or downloaded
            prebuilt when available.

            ``variant="fate_prediction"`` is a deprecated alias for
            ``"unprocessed"`` and will be removed in a future release. Despite its
            upstream filename (``...in_vitro.gene_filtered.h5ad``), that object is
            the unfiltered one.
        filter_genes: bool, default=True
            Whether to subset to ``adata.var['use_genes']``. A no-op for variants
            that are already gene-filtered.
        reduce_dimensions: bool, default=True
            Whether to scale and run PCA (50 components, ``random_state=0``).
        cytotrace: bool, default=True
            Whether to annotate with precomputed CytoTRACE values. Applied
            independently of the other preprocessing flags.
        force_download: bool, default=False
            Re-download the raw file and regenerate the processed file.
        force_preprocess: bool, default=False
            Regenerate the processed file from the cached raw file without
            re-downloading it.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    variant = _resolve_variant(variant)
    data_handler = LARRYInVitroDataset(
        data_dir=data_dir,
        variant=variant,
        filter_genes=filter_genes,
        reduce_dimensions=reduce_dimensions,
        cytotrace=cytotrace,
        force_download=force_download,
        force_preprocess=force_preprocess,
    )
    return data_handler.adata
