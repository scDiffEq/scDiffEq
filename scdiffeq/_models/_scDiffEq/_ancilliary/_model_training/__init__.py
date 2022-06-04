
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


from ._define_training_program import _define_training_program as define_training_program
from ._execute_training_program import _execute_training_program as execute_training_program

from ._Learner import _Learner as Learner