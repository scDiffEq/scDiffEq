
__module_name__ = "_CustomModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import local dependencies #
# ------------------------- #
from ._core._BaseModel import BaseModel


# ------------------------------------------------------------------------------------- #
# MAIN MODULE CLASS: CustomModel
# ------------------------------------------------------------------------------------- #

class CustomModel(BaseModel):
    
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.model_setup(dataset, func)
        
    def fit(self):
        
        ...
        
    def evaluate(self):
        
        ...
        
# ------------------------------------------------------------------------------------- #