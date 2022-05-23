# Tools

#### PCA

```python
sdq.tl.pca(adata)
```

#### UMAP

```python
sdq.tl.umap(adata)
```

#### Quantify the relative contribution of the drift and diffusion functions

```python
sdq.tl.quantify_dynamics(model)
```

#### Generatively simulate data

This function is identical to `scDiffEq.simulate`

Optionally, this returns a `simulation` object, which can then be studied using other ancilliary functions such as `sdq.tl.pca`, `sdq.tl.umap`, `sdq.tl.quantify_dynamics`, and visualized  using `sdq.pl.pca`, `sdq.pl.umap`, `sdq.pl.dynamics`, `sdq.pl.drift`, or `sdq.pl.diffusion`.

```python
simulation = sdq.tl.simulate(model, X)
```