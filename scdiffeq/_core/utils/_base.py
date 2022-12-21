from abc import ABC


class Base(ABC):
    """Base class for automatic parsing of args."""

    def __parse__(self, kwargs, ignore=["self", "kwargs"], hide=[]):

        self._PASSED_KWARGS = {}
        for key, val in kwargs.items():
            if not key in ignore:
                self._PASSED_KWARGS[key] = val
                if key in hide:
                    key = "_{}".format(key)
                setattr(self, key, val)
            elif key == "kwargs":
                self._PASSED_KWARGS = self._split_kwargs(val, self._PASSED_KWARGS)

    def _split_kwargs(self, kw, KWARG_DICT={}):
        for k, v in kw.items():
            KWARG_DICT[k] = v
            setattr(self, k, v)
        return KWARG_DICT