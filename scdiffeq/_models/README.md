# Models

```python
import scdiffeq as sdq
```

The organizational goal of this module is to give a user/developer the ability to setup models in four ways:

1. The primary model we propose: `scDiffEq`

```python
model = sdq.models.scDiffEq()
model.fit()
```

2. The contemporary model: `PRESCIENT` implemented using pytorch-lightning.

```python
model = sdq.models.PRESCIENT()
model.fit()
```

3. A custom model class that can be defined through arguments passed through: 

```python
model = sdq.models.build(**model_kwargs)
model.fit()
```

4. The base class, named `BaseModel`, from which all other models may be constructed. This is the class that is called underneath the other three.

```python
model = sdq.models.build(**model_kwargs)
model.fit()
```