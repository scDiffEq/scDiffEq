

import yaml
import ABCParse

class HyperParams(ABCParse.ABCParse):
    def __init__(self, yaml_path):

        self.__configure__(locals())

    def _read(self):
        if not hasattr(self, "_file"):
            self._yaml_file = yaml.load(open(self._yaml_path), Loader = yaml.Loader)

    def __configure__(self, kwargs, private=["yaml_path"]):

        self.__parse__(kwargs, private=private)
        self._read()
        for key, val in self._yaml_file.items():
            setattr(self, key, val)
            
    @property
    def attrs(self):
        self._attrs = {attr: getattr(self, attr) for attr in self.__dir__() if not attr[0] in ["_", "a"]}  
        return self._attrs
    
    def __repr__(self):
        
        """Return a readable representation of the discovered hyperparameters"""
        
        
        string = "HyperParameters\n"
        for attr, val in self.attrs.items():
            string += "\n  {:<34}: {}".format(attr, val)
            
        return string