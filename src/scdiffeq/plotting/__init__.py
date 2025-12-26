
__module_name__ = "__init__.py"
__doc__ = """plotting __init__ module. Sub-package of the main scdiffeq API."""


# import functions accessed as sdq.pl.<func>: --------------------------------------------
from ._velocity_stream import velocity_stream
from ._temporal_expression import temporal_expression
from ._simulation_umap import simulation_umap
from ._simulation_trajectory_gif import simulation_trajectory_gif
