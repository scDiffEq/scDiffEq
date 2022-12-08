
from ._base_utility_functions import extract_func_kwargs

# -- Main module class: ------------------------------------------------------------------
class Parser:
    _kwargs = {}
    _literal_kwargs = {}

    def __self_parse__(self, kwargs, ignore=["self", "kwargs", "__class__"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)

    def __init__(self, obj, passed_kwargs, ignore=["self", "kwargs", "__class__"], hide=[]):

        keys = list(passed_kwargs.keys())
        self.__self_parse__(locals())

    def literal_kwargs(self):
        """parse an argument ductionary that is literally passed with the keyword name: kwargs"""
        if "kwargs" in self.keys:
            for key, val in self.passed_kwargs["kwargs"].items():
                setattr(self.obj, key, val)
                self._literal_kwargs[key] = val

    def kwargs(self):
        """parse all kwargs. this is the normal use-case function"""
        for key, val in self.passed_kwargs.items():
            if not key in self.ignore:
                setattr(self.obj, key, val)
                self._kwargs[key] = val
                
    def func_specific_kwargs(self, func):
        """Look for additional kwargs given some function."""
        
        function_kwargs = {}
        existing_kwargs = list(self._kwargs.keys())
        func_kwargs = extract_func_kwargs(func=func, kwargs=self._literal_kwargs)
        for key, val in func_kwargs.items():
            if not key in existing_kwargs:
                function_kwargs[key] = val
                
        return function_kwargs

    def __call__(self):
        self.kwargs()
        self.literal_kwargs()

        
# -- one-line API-facing function: -------------------------------------------------------
def parser(obj, kwargs, func=None):
    
    p = Parser(obj, kwargs)
    p()
    
    if func:
        func_kwargs = p.func_specific_kwargs(func)
        return p._kwargs, func_kwargs
    
    return p._kwargs