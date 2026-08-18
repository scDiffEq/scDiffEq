# Changelog

## Unreleased

### Training no longer crashes in a notebook on a multi-GPU machine

`fit` in a Jupyter or Colab kernel on a host with more than one GPU raised `RuntimeError: Cannot re-initialize CUDA in forked subprocess` ([#111](https://github.com/mvinyard/scDiffEq/issues/111)).

The trainer configuration overwrote whatever `devices` the caller passed with `torch.cuda.device_count()`, so any multi-GPU host selected DDP. Lightning launches DDP from a notebook by forking the kernel, and a forked process cannot re-initialize CUDA. The number of GPUs on the machine therefore decided whether the quickstart ran at all.

`devices` is now honored as passed. Multi-device training is capped to one device inside a notebook, with a warning saying so and pointing at running as a script. Lightning's fork-based launcher cannot do it. Outside a notebook the previous default is unchanged: `devices=None` still uses (and logs) all visible CUDA devices. Pin with `fit(devices=1)`.

### Fixed

- `devices` passed to `fit`, `train` or `pretrain` was overwritten by the GPU count, so a single-GPU run could not be requested on a multi-GPU host.
- `pretrain()` raised `TypeError: 'devices' selected with 'CPUAccelerator' should be an int > 0` on any machine without CUDA, since its documented `devices=None` default reached the Trainer unresolved.

## 1.1.3 (2026-08-17)

### Fixed

- A `weight_key` naming a column that is not in `adata.obs` now warns instead of silently filling it with 1 and training unweighted.
- `obs_keys` no longer leaks the weight key into models built later in the same session.
- `tools.simulate` now raises when the time span read from `adata` is a single point, instead of returning the input unintegrated.
- The `sim` trajectory id is now separator-delimited, so `z0_idx` and `sim_i` cannot concatenate into a key shared by two trajectories.

## 1.1.2 (2026-08-16)

### Declare the dependencies scdiffeq imports

`numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `pyyaml` and `plotly` are imported at module scope but were never declared. They arrived transitively via scanpy, anndata and lightning. Declared to prevent a break.

Updated to `pydk 0.0.55`, which fixes a broken `pytz` import.

`plotly` was broken: `scdiffeq.plotting.potential_landscape` could not be imported after a clean install, since nothing else in the dependency tree provides plotly.

### Added

- An `optional` extra for integrations that are imported lazily, inside the functions that use them, so the package works without them until you use the corresponding feature: `umap-learn`, `pillow`, `ipython`, `psutil`, `wandb`.

## 1.1.1 (2026-08-15)

### Dataset downloads are working again

The Figshare download host began answering programmatic requests with an HTTP 202 web-application-firewall challenge and an empty body, which broke every dataset download. Downloads are now served from a Zenodo record and verified against its published md5 checksums:

**DOI: [10.5281/zenodo.21947161](https://doi.org/10.5281/zenodo.21947161)**

Figshare remains as a fallback via API endpoint rather than the blocked download host. A download that returns a challenge page, truncated body, or anything lacking the expected file signature is now rejected before it reaches the cache instead of failing later inside `anndata.read_h5ad`.

### Breaking and behavioural changes

- **`larry(variant="fate_prediction")` is deprecated**; use `variant="unprocessed"`. The old name still works and warns.

  Rename corrects a misleading name. Despite its upstream filename (`adata.Weinreb2020.in_vitro.gene_filtered.h5ad`), that object is the *unfiltered* input: 130,887 × 25,289 with no `X_pca` and a `use_genes` column to filter by. The default variant is the already-filtered one: 130,887 cells × 2,492 genes with a precomputed .obsm[`X_pca`].

- **Cache layout changed.** The raw download and the preprocessed result are now separate files (`_larry.raw.h5ad` and `larry.processed.h5ad`) rather than sharing one name. Existing caches are detected and renamed into the new layout automatically. Large, multi-GB files are not re-downloaded.

- **PCA is seeded** (`random_state=0`) in the `larry`, `human_hematopoiesis`, and `pancreatic_endocrinogenesis` loaders.

  This changes results relative to previous versions. Dimension reduction previously used an unseeded randomized SVD, so two runs on identical input produced different components. Results are now reproducible **going forward**, but the seeded basis does **not** reproduce the one distributed with the published analyses, which exists now only as a stored array. To use that exact basis, users should use the `X_pca` shipped inside the dataset.

- **`ipykernel` removed as a runtime dependency.** Moved to the `dev` extra; a plain install no longer pulls Jupyter machinery. `scikit-learn` and `h5py` are now declared explicitly. Both were already imported directly but only satisfied transitively.

### Fixed

- Preprocessing ran only on the download path, so an `.h5ad` already present on disk was returned raw, with no `X_pca`. Preprocessing is now driven by the processed file's absence or invalidity, so a dataset obtained by any route, including a manual download, is preprocessed on first use.
- The raw download and preprocessed result shared one path, so a preprocess that failed partway left a raw file where the processed one belonged and every later call silently returned it. Writes are now atomic, and a cached processed file is structurally validated and regenerated if invalid.
- `LARRYInVitroDataset.adata` returned `None` on its second access.
- The `cytotrace` flag was ignored unless `filter_genes` or `reduce_dimensions` was also set, and its two annotation CSVs were re-downloaded on every call.
- CytoTRACE annotations were merged with `pd.concat(axis=1)`, which took the union of the indexes: it lengthened `obs`/`var` when annotation keys were absent from the target, and duplicated columns when run twice. Partially covered annotations are now joined correctly, and a partially covered boolean column no longer produces an object column that cannot be written to `.h5ad`.
- Fitted `scaler`/`pca` models from different variants overwrote each other. The default variant keeps the documented `scaler.pkl` / `pca.pkl` names; others are namespaced.
- Non-default preprocessing flags now get their own cache file, so `reduce_dimensions=False` no longer poisons the default cache.
- `pancreatic_endocrinogenesis` passed a `url_prefix` argument the downloader does not accept, raising `TypeError`.

### Added

- `larry(force_preprocess=True)` regenerates the processed file from the cached raw file without re-downloading it.
- A test suite (`pytest`, new `test` extra) and a CI workflow that runs it on pull requests. Network-dependent tests are marked `slow` and excluded by default.
- `scripts/mirror_to_zenodo.py`, the maintainer tool that performs the Zenodo migration.
