
from abc import ABC

class Base(ABC):
    _KWARGS = {}

    def __init__(self):
        pass

    def __hide__(self, key):
        return "_{}".format(key)

    def __collect__(self, key, val):
        self._KWARGS[key] = val
        setattr(self, key, val)

    def __parse__(
        self,
        kwargs: dict,
        ignore: list = ["self"],
        private: list = [],
        public: list = [],
    ):

        """
        Parameters:
        -----------

        Notes:
        ------
        (1) assumes all are public unless denoted in private
        (2) If a public list is provided, all kwargs are shifted to private unless denoted in public.
        """

        if len(public) > 0:
            private = list(kwargs.keys())

        for key, val in kwargs.items():
            if not key in ignore:
                if (key in private) and (not key in public):
                    key = self.__hide__(key)
                self.__collect__(key, val)
