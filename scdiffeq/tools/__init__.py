
# machine learning __init__.py

__author__ = ', '.join([
    'Michael E. Vinyard'
])
__email__ = ', '.join([
    'vinyard@g.harvard.edu',
])

from .sc_odeint import sc_odeint
from .get_minibatch import get_minibatch

from ._diffeq_funcs import ode_func as ode
from ._diffeq_funcs import sde_func as sde

from .train_model import train_model

from ._pca import pca

from .evaluate import evaluate_test_traj as evaluate