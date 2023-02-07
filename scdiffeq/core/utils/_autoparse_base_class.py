

# -- import packages: --------------------------------------------------------------------
from abc import ABC
import inspect


# -- Main class: -------------------------------------------------------------------------
class AutoParseBase(ABC):

    def __init__(self):
        self._KWARGS = {}
    
    def __call__(self):
        pass
    
    def __init_kwargs__(self):
        self._KWARGS = {}
        
    def _inspect(self, func):
        return list(inspect.signature(func).parameters.keys())
        
    @property
    def _init_params(self):
        return self._inspect(func=self.__init__)
    
    @property
    def _call_params(self):
        return self._inspect(func=self.__call__)
    
    @property
    def _parse_params(self):
        return self._inspect(func=self.__parse__)

    @property
    def _collected_params(self):
        return self._init_params + self._call_params + self._parse_params        

    def _collect_literal_kwargs(self, kwargs_val):
        
        for key, val in kwargs_val.items():
            self.__collect__(key, val)
    
    def __hide__(self, key):
        return "_{}".format(key)

    def __collect__(self, key, val):
        if not hasattr(self, "_KWARGS"):
            self.__init_kwargs__()
            
        self._KWARGS[key] = val
        setattr(self, key, val)

    def __parse__(
        self,
        kwargs:  dict,
        ignore:  list = ["self"],
        private: list = ["ignore", "private", "public"],
        public:  list = [],
        kwargs_key: str = "kwargs",
    ):
        """
        Pass `locals()` or some other collection of kwargs to `kwargs` to
        save them as attributes of the subclass.
        
        Parameters:
        -----------
        kwargs
            typically `locals()`
            type: dict
        
        ignore
            type: list
            default: ["self"]
        
        private
            type: list
            default: ["ignore", "private", "public"]
        
        public
            type: list
            default: []
        
        kwargs_key
            type: str
            default: "kwargs"

        Notes:
        ------
        (1) assumes all are public unless denoted in private
        (2) If a public list is provided, all kwargs are shifted to private unless denoted in public.
        """

        if len(public) > 0:
            private = list(kwargs.keys())

        for key, val in kwargs.items():
            if not key in ignore: # (key in self._collected_params) and ():
                if key == kwargs_key:
                    self._collect_literal_kwargs(val)
                else:
                    if (key in private) and (not key in public):
                        key = self.__hide__(key)
                    self.__collect__(key, val)
