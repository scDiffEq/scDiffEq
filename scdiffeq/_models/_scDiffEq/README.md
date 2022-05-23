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
