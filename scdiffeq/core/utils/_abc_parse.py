from abc import ABC
from typing import Dict, List, Any


class ABCParse(ABC):
    BUILT = False

    def __init__(self, *args, **kwargs):
        """
        we avoid defining things in __init__ because this subsequently
        mandates the use of `super().__init__()`
        """
        pass

    def __build__(self) -> None:

        self._PARAMS = {}
        self._IGNORE = ["self", "__class__"]
        self._stored_private = []
        self._stored_public = []

        self.BUILT = True

    def __set__(
        self, key: str, val: Any, public: List = [], private: List = []
    ) -> None:

        self._PARAMS[key] = val
        if (key in private) and (not key in public):
            self._stored_private.append(key)
            key = f"_{key}"
        else:
            self._stored_public.append(key)
        setattr(self, key, val)

    def __set_existing__(self, key: str, val: Any) -> None:

        self._PARAMS[key] = val

        if key in self._stored_private:
            key = f"_{key}"
        setattr(self, key, val)

    @property
    def _STORED(self) -> List:
        return self._stored_private + self._stored_public

    def __parse__(
        self, kwargs: Dict, public: List = [], private: List = [], ignore: List = []
    ):

        if not self.BUILT:
            self.__build__()

        self._IGNORE += ignore

        if len(public) > 0:
            private = list(kwargs.keys())

        for key, val in kwargs.items():
            if not key in self._IGNORE:
                self.__set__(key, val, public, private)

    def __update__(self, kwargs: dict, public: List = [], private: List = []) -> None:

        if not self.BUILT:
            self.__build__()

        for key, val in kwargs.items():
            if not isinstance(val, NoneType) and key in self._STORED:
                self.__set_existing__(key, val)
            elif not isinstance(val, NoneType) and not key in self._IGNORE:
                self.__set__(key, val, public, private)