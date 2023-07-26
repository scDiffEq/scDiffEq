from typing import Dict, List, Any, Tuple
from abc import ABC


class ABCParse(ABC):
    _BUILT = False

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

        self._BUILT = True

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
        
        passed_key = key

        if key in self._stored_private:
            key = f"_{key}"

        if passed_key == "kwargs":
            attr = getattr(self, key)
            attr.update(val)
            setattr(self, key, attr)
            self._PARAMS.update(val)
            
        elif passed_key == "args":
            attr = getattr(self, key)
            attr += val
            setattr(self, key, attr)
            self._PARAMS[passed_key] += val
            
        else:
            self._PARAMS[passed_key] = val
            setattr(self, key, val)

    @property
    def _STORED(self) -> List:
        return self._stored_private + self._stored_public

    def __setup_inputs__(self, kwargs, public, private, ignore) -> Tuple[List]:
        if not self._BUILT:
            self.__build__()

        self._IGNORE += ignore

        if len(public) > 0:
            private = list(kwargs.keys())

        return public, private

    def __parse__(
        self, kwargs: Dict, public: List = [], private: List = [], ignore: List = []
    ) -> None:
        """Central function of this class"""

        public, private = self.__setup_inputs__(kwargs, public, private, ignore)

        for key, val in kwargs.items():
            if not key in self._IGNORE:
                self.__set__(key, val, public, private)

    def __update__(
        self, kwargs: Dict, public: List = [], private: List = [], ignore: List = []
    ) -> None:
        """To be called after __parse__ has already been called."""

        public, private = self.__setup_inputs__(kwargs, public, private, ignore)

        for key, val in kwargs.items():
            if not (val is None) and (key in self._STORED):
                self.__set_existing__(key, val)

            elif not (val is None) and not (key in self._IGNORE):
                self.__set__(key, val, public, private)