
# _evaluate_diffeq.py
__module_name__ = "_evaluate_diffeq.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
import licorice
import time


# local imports #
# ------------- #
# from ._plot_evaluation import _plot_evaluation
from .._common_functions._IntegratorModule import _Integrator


def _evaluate_diffeq(DiffEq, n_batches, device, plot, plot_save_path, use, time_key, plot_title_fontsize):

    """"""
        
    evaluator = _Integrator(mode  = 'test',
                            network_model = DiffEq.network_model,
                            device=device,
                            diffusion = DiffEq.diffusion,
                            integration_function = DiffEq.integration_function,
                            HyperParameters = DiffEq.hyper_parameters,
                            TrainingMonitor = DiffEq.TrainingMonitor,
                            use=use, 
                            time_key=time_key,
    )
    
    batch_start = time.time()
    print("Creating Batches....", end = "\r")
    evaluator.batch_data(DiffEq.adata, n_batches)
    print("Creating Batches.... {:.3f}s elapsed.".format(time.time() - batch_start))
    
    int_start = time.time()
    print("Forward integration.... ", end = "\r")
    evaluator.forward_integrate()
    print("Forward integration.... {:.3f}s elapsed.".format(time.time() - int_start))

    loss_start = time.time()
    print("Loss calculation.... ", end = "\r")
    evaluator.calculate_loss()
    print("Loss calculation.... {:.3f}s elapsed.".format(time.time() - loss_start))
    
    
    plot_start = time.time()
    print("Plotting.... ", end = "\r")
#     if plot:
#         _plot_evaluation(evaluator, 
#                          title_fontsize=plot_title_fontsize, 
#                          save_path=plot_save_path, 
#                          TrainingMonitor=DiffEq.TrainingMonitor)
    print("Plotting.... {:.3f}s elapsed.".format(time.time() - plot_start))
    msg = licorice.font_format("Test loss:", ['BOLD', 'CYAN'])
    print("{} {:.6f}".format(msg, evaluator.test_loss))
    

    return evaluator