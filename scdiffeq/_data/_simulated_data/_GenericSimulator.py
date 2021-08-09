
# package imports #
# --------------- #
import vintools as v

# local imports #
# ------------- #
from ._simulation_supporting_functions._get_initial_conditions import _get_initial_conditions
from ._simulation_supporting_functions._get_numpy_distribution_function import _get_numpy_distribution_function
from ._simulation_supporting_functions._print_verbose_function_documentation import _print_verbose_function_documentation
from ._simulation_supporting_functions._create_time_vector import _create_time_vector
from ._simulation_supporting_functions._formatting_simulation_for_AnnData import _simulation_to_AnnData
from ._simulation_supporting_functions._plot_simulation import _plot

# # import pre-defined state functions #
# # ---------------------------------- #
from . import _simulation_state_equations as _sim_eqns

class _GenericSimulator:
    def __init__(
        self,
    ):
        
        """
        Notes:
        ------
        (1) Right now, this function / class is limited by the static import of these state equations (directly
            below). I want to find a way to dynamically import these such that I can more easily add more, but I 
            have yet to find a way to do so. 
        """
        
        print("Simulator initiated. Importing preloaded state functions.\n")
                
        self.StateFuncEquationDict = {}
        self.StateFuncEquationDict['parabola_2d'] = _sim_eqns._parabola_2d_state_equation
        self.StateFuncEquationDict['four_attractor_2d'] = _sim_eqns._four_attractor_2d_state_equation
            
    def set_initial_conditions_sampling_distribution(
        self,
        function,
        module="random",
        package="numpy",
        print_verbose_documentation=False,
    ):

        """Set the sampling distrubution of the initial conditions for the simulation."""
        self.distrubution_function = _get_numpy_distribution_function(
            function,
            module=module,
            package=package,
            print_verbose_documentation=False,
        )
        func = ".".join([package, module, function])
        
        self.package = package
        self.module = module
        self.function = function
        
        print(
            "{} {}".format(
                v.ut.format_pystring("Initial conditions sampling function:", ["BOLD"]),
                v.ut.format_pystring(func, ["BOLD", "RED"]),
            )
        )
        
    def print_distribution_function_docs(
        self,
    ):

        """"""

        _print_verbose_function_documentation(
            self.distrubution_function, self.package, self.module, self.function
        )
    def get_initial_conditions(self, plot=False, n_bins=20, **kwargs):
        
        self.initial_conditions = _get_initial_conditions(
            self.distrubution_function, **kwargs
        )
        self.n_traj = len(self.initial_conditions)
        
        if plot:
            v.pl.hist2d_component_plot(data=self.initial_conditions, n_bins=n_bins, suptitle="Initial conditions")
    
    
    def create_time_vector(self, time_span=10.0, n_samples=1000, noise_amplitude=0):
        self.time_vector = _create_time_vector(time_span=time_span, n_samples=n_samples, noise_amplitude=noise_amplitude)
    
    def simulate_ODE(self, state_function, to_adata=True, plot=True, **kwargs):
        
        """"""
        
        from scipy.integrate import odeint as scipy_odeint
        
        assert state_function in self.StateFuncEquationDict.keys(), print(
            "\nPlease choose from: {}".format(
                v.ut.format_pystring(self.StateFuncEquationDict.keys(), ['BOLD'])
            )
        )    
        
        func = self.StateFuncEquationDict[state_function]
        self.simulated_trajectories = []
        
        print("\nSimulating {} trajectories from {} over {} time points in each trajectory...".format(self.n_traj, state_function, len(self.time_vector)))
        
        for y0 in self.initial_conditions:
            trajectory = scipy_odeint(func=func, y0=y0, t=self.time_vector)            
            self.simulated_trajectories.append(trajectory)
        
        if to_adata:
            self.adata = _simulation_to_AnnData(self, silent=False)
        
        if plot:
            _plot(self, c='time', s=12, alpha=0.5, **kwargs)
            
    def plot_simulation(self, c='time', s=12, alpha=0.5, **kwargs):

        _plot(self, c=c, s=s, alpha=alpha, **kwargs)
        
    def plot_initial_conditions(self, n_bins=20, **kwargs):

        v.pl.hist2d_component_plot(data=self.initial_conditions, n_bins=n_bins, suptitle="Initial conditions")
    