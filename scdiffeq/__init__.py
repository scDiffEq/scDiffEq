# scdiffeq __init__.py

__author__ = ', '.join([
    'Michael E. Vinyard'
])
__email__ = ', '.join([
    'vinyard@g.harvard.edu',
])

<<<<<<< HEAD
from . import _data as data
from . import _tools as tl
from . import _plotting  as pl
from . import _utilities as ut
=======
from . import data
from . import tools as tl
from . import plotting  as pl
from . import utilities as ut

import pandas as pd
pd.set_option('display.max_columns', None)
>>>>>>> 93f9d6c2d5aece01bdf92b3c286e1fd2a7107a32
