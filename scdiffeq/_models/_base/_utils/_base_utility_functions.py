
__module_name__ = "_base_utility_functions.py"
__doc__ = """To-do"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.44"


# -- import packages: --------------------------------------------------------------------
import inspect
import torch


# -- functions: --------------------------------------------------------------------------
# def autodevice():
#     if torch.cuda.is_available():
#         return torch.device("cuda:{}".format(torch.cuda.current_device()))
#     return torch.device("cpu")


def func_params(func):
    return list(inspect.signature(func).parameters.keys())


def extract_func_kwargs(func, kwargs):
    func_kwargs = {}
    params = func_params(func)
    for k, v in kwargs.items():
        if k in params:
            func_kwargs[k] = v
    return func_kwargs

def local_arg_parser(kwargs, ignore=["self"]):

    parsed_kwargs = {}
    for k, v in kwargs.items():
        if not k in ignore:
            parsed_kwargs[k] = v

    return parsed_kwargs
