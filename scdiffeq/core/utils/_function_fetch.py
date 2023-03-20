
# -- import packages: ----------------------------------------------------------


# -- import local dependencies: ------------------------------------------------
from ._autoparse_base_class import AutoParseBase


# -- define types: -------------------------------------------------------------
from typing import Union, Any
NoneType = type(None)


class FunctionFetch(AutoParseBase):
    """Fetch a function from a flexible input."""
    def __init__(self, module=None, parent=None):
        self.__parse__(locals())

    def __call__(self, func: Union[Any, str]):
        if isinstance(func, str):
            return getattr(self.module, func)
        elif issubclass(func, self.parent):
            return func
        else:
            print("pass something else....")