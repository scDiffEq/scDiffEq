
import os

class _PackageVersion:
    
    def __init__(self):
        ...
        
    @property
    def PACKAGE_PATH(self):
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    @property
    def SETUP_FPATH(self):
        return os.path.join(self.PACKAGE_PATH, "setup.py")
    
    def _read_setup_dot_py(self):
        f = open(self.SETUP_FPATH)
        file = f.readlines()
        f.close()
        return file
    
    @property
    def VERSION(self):
        file = self._read_setup_dot_py()
        return [line for line in file if "version" in line][0].split("version=")[1].split('"')[1]
    
    def __call__(self):
        return self.VERSION
