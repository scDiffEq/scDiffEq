# Changelog

## 1.1.1 (2026-08-15)

### Dataset downloads are working again

The Figshare download host began answering programmatic requests with an HTTP 202
web-application-firewall challenge and an empty body, which broke every dataset
download. Downloads are now served from a Zenodo record and verified against its
published md5 checksums:

**DOI: [10.5281/zenodo.21947161](https://doi.org/10.5281/zenodo.21947161)**

Figshare remains in the cascade as a fallback, via its API endpoint rather than
the blocked download host. A download that returns a challenge page, a truncated
body, or anything lacking the expected file signature is now rejected before it
reaches the cache instead of failing later inside `anndata.read_h5ad`.

### Breaking and behavioural changes

- **`larry(variant="fate_prediction")` is deprecated**; use
  `variant="unprocessed"`. The old name still works and warns.

  The rename corrects a misleading name. Despite its upstream filename
  (`adata.Weinreb2020.in_vitro.gene_filtered.h5ad`), that object is the
  *unfiltered* input: 130,887 × 25,289 with no `X_pca` and a `use_genes` column to
  filter by. The default variant is the already-filtered one (130,887 × 2,492)
  that ships a precomputed `X_pca`.

- **Cache layout changed.** The raw download and the preprocessed result are now
  separate files (`_larry.raw.h5ad` and `larry.processed.h5ad`) rather than
  sharing one name. Existing caches are detected and renamed into the new layout
  automatically — multi-gigabyte files are not re-downloaded.

- **PCA is seeded** (`random_state=0`) in the `larry`, `human_hematopoiesis`, and
  `pancreatic_endocrinogenesis` loaders.

  > **This changes results relative to previous versions.** Dimension reduction
  > previously used an unseeded randomized SVD, so two runs on identical input
  > produced different components. Results are now reproducible **going forward**,
  > but the seeded basis does **not** reproduce the one distributed with the
  > published analyses, which exists now only as a stored array. If you need that
  > exact basis, use the `X_pca` shipped inside the dataset rather than
  > recomputing it.

- **`ipykernel` is no longer a runtime dependency.** It moved to the `dev` extra;
  a plain install no longer pulls Jupyter machinery. `scikit-learn` and `h5py` are
  now declared explicitly — both were already imported directly but only satisfied
  transitively.

### Fixed

- Preprocessing ran only on the download path, so an `.h5ad` already present on
  disk was returned raw, with no `X_pca`. Preprocessing is now driven by the
  processed file's absence or invalidity, so a dataset obtained by any route
  — including a manual download — is preprocessed on first use.
- The raw download and preprocessed result shared one path, so a preprocess that
  failed partway left a raw file where the processed one belonged and every later
  call silently returned it. Writes are now atomic, and a cached processed file is
  structurally validated (checked from HDF5 metadata, so it costs a few KB even
  for a 5 GB file) and regenerated if invalid.
- `LARRYInVitroDataset.adata` returned `None` on its second access.
- The `cytotrace` flag was ignored unless `filter_genes` or `reduce_dimensions`
  was also set, and its two annotation CSVs were re-downloaded on every call.
- CytoTRACE annotations were merged with `pd.concat(axis=1)`, which took the union
  of the indexes: it lengthened `obs`/`var` when annotation keys were absent from
  the target, and duplicated columns when run twice. Partially covered annotations
  are now joined correctly, and a partially covered boolean column no longer
  produces an object column that cannot be written to `.h5ad`.
- Fitted `scaler`/`pca` models from different variants overwrote each other. The
  default variant keeps the documented `scaler.pkl` / `pca.pkl` names; others are
  namespaced.
- Non-default preprocessing flags now get their own cache file, so
  `reduce_dimensions=False` no longer poisons the default cache.
- `pancreatic_endocrinogenesis` passed a `url_prefix` argument the downloader does
  not accept, raising `TypeError`.

### Added

- `larry(force_preprocess=True)` regenerates the processed file from the cached
  raw file without re-downloading it.
- A test suite (`pytest`, new `test` extra) and a CI workflow that runs it on pull
  requests. Network-dependent tests are marked `slow` and excluded by default.
- `scripts/mirror_to_zenodo.py`, the maintainer tool that performs the Zenodo
  migration.
