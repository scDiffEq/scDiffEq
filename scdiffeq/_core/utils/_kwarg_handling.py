
# -- import packages: --------------------------------------------------------------------
import inspect
import torch


# -- functions: --------------------------------------------------------------------------
def func_params(func):
    return list(inspect.signature(func).parameters.keys())


def extract_func_kwargs(func, kwargs, ignore=[]):
    func_kwargs = {}
    params = func_params(func)
    for k, v in kwargs.items():
        if not k in ignore:
            if k in params:
                func_kwargs[k] = v
    return func_kwargs
