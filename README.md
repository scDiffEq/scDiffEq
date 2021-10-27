# ![scdiffeq-logo](docs/imgs/scdiffeq.logo.svg)

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scdiffeq.svg)](https://pypi.python.org/pypi/scdiffeq/)
[![PyPI version](https://badge.fury.io/py/scdiffeq.svg)](https://badge.fury.io/py/scdiffeq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An analysis framework for modeling dynamical single-cell data with **stochastic neural differential equations**.

## Install the development package:

```
git clone git@github.com:pinellolab/sc-neural-ODEs.git

pip install -e .
```

## General workflow:

**1**. Instantiate a neural differential equation compatible with scRNA-seq data formatted as [**`AnnData`**](https://anndata.readthedocs.io/en/stable/).
```python
import scdiffeq as sdq

DiffEq = sdq.scDiffEq()
```

**2**. Pass **`adata`** to the model and define any "preflight" parameters. **`adata`** is split between **test** / **train** / **validation** sets.
```python
DiffEq.preflight(adata)
```

**3**. Forward integrate through the dataset to learn the dynamics governing the latent (or passed) variable.
```python
DiffEq.learn()
```

**4**. Evaluate the model with the partitioned test data.
```python
DiffEq.evaluate()
```

**5**. Compute and interpret the [quasi-potential landscape](https://royalsocietypublishing.org/doi/10.1098/rsif.2012.0434)<sup>**1**</sup>.
```python
DiffEq.compute_landscape()
```

---

**References**:

**1**. [Zhou Joseph Xu, _et al_](https://scholar.google.com/citations?user=cMBBPisAAAAJ&hl=en)., [Quasi-potential landscape in complex multi-stable systems](https://royalsocietypublishing.org/doi/10.1098/rsif.2012.0434?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed). *J. R. Soc. Interface*. (**2012**). [DOI](http://doi.org/10.1098/rsif.2012.0434).
