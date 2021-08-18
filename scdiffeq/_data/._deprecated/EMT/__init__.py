# EMT simulation __init__.py

__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


from .run_emt import simulate_iteratively as simulate
from .plot_EMT import plot_EMH
from .utils import grab_stable_state_conditions as get_ss
