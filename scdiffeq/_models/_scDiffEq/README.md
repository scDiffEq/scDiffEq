# scDiffEq model

#### Instantiate the main model class
```python
model = sdq.models.scDiffEq(adata)
```

#### Define / verify the model, data, and the training program 
```python
model.preflight()
```

#### Subject the model to the planned training program

```python
model.train()
```

#### Generatively simulate data

Optionally, this returns a `simulation` object, which can then be studied using other ancilliary functions such as `sdq.tl.pca`, `sdq.tl.umap`, `sdq.tl.quantify_dynamics`, and visualized  using `sdq.pl.pca`, `sdq.pl.umap`, `sdq.pl.dynamics`, `sdq.pl.drift`, or `sdq.pl.diffusion`.

This function is identical to `sdq.tl.simulate`

```python
simulation = model.simulate()
```

#### Pass data to the model independent of the training program.

```python
model.evaluate()
```

#### Save a model checkpoint

```python
model.save()
```

#### Load a previous model checkpoint
Loads the model without necessarily loading the entire run associated with that model.
```python
model.load_model()
```

#### Load a previous model run
Loads the entire model along with the ancilliary metadata and saved sub-class components of the model (i.e., `model._Learner` and `model._ModelManager`). This is useful for continuation of training.
```python
model.load_run()
```
