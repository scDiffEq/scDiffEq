# scdiffeq __init__.py

__author__ = ', '.join([
    'Michael E. Vinyard'
])
__email__ = ', '.join([
    'vinyard@g.harvard.edu',
])


from . import _data as data
from . import _tools as tl
from . import _plotting  as pl
from . import _utilities as ut

import pandas as pd
pd.set_option('display.max_columns', None)
