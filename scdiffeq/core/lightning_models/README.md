
### Some implementation notes

1. Upgraded to torch 2.0 and lightning 2.0. This required several behind-the-scenes changes to the original API.
2. We don't necessarily need to have access to a lone `TorchNet` as before since the `ODE` is now properly implemented and accessible. 


### Still to-do

1. Integrate callback(s) for saving pre-train loss and loss chart(s)
2. Automatic adjustment of SDE/ODE parameters for all models.
3. Set "automatic" pre-defined params to be closer to ideal values.

### Tests

This is a good test to make sure the default args don't throw errors for the instantiation of each model:

```python
import scdiffeq as sdq

mods = [mod for mod in sdq.core.lightning_models.__dir__() if mod.startswith("Lightning")]
kwargs = {"data_dim": 2447, "latent_dim": 20}

for mod in mods:
    print(mod, end="-")
    model = getattr(sdq.core.lightning_models, mod)
    kw = sdq.core.utils.extract_func_kwargs(func = model, kwargs = kwargs)
    print(model(**kw))
```