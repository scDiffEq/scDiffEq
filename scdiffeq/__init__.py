# scdiffeq __init__.py

__author__ = ', '.join([
    'Michael E. Vinyard'
])
__email__ = ', '.join([
    'vinyard@g.harvard.edu',
])

from . import data
from . import tools as tl
from . import plotting  as pl
from . import utilities as ut

import pandas as pd
pd.set_option('display.max_columns', None)