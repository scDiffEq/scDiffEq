


# -- Main module class: ------------------------------------------------------------------
class Parser:
    _kwargs = {}

    def __self_parse__(self, kwargs, ignore=["self", "kwargs"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)

    def __init__(self, obj, passed_kwargs, ignore=["self", "kwargs"], hide=[]):

        keys = list(passed_kwargs.keys())
        self.__self_parse__(locals())

    def literal_kwargs(self):
        """parse an argument ductionary that is literally passed with the keyword name: kwargs"""
        if "kwargs" in self.keys:
            for key, val in self.passed_kwargs["kwargs"].items():
                setattr(self.obj, key, val)
                self._kwargs[key] = val

    def kwargs(self):
        """parse all kwargs. this is the normal use-case function"""
        for key, val in self.passed_kwargs.items():
            if not key in self.ignore:
                setattr(self.obj, key, val)
                self._kwargs[key] = val

    def __call__(self):
        self.kwargs()
        self.literal_kwargs()

        
# -- one-line API-facing function: -------------------------------------------------------
def parser(obj, kwargs):
    
    p = Parser(obj, kwargs)
    p()