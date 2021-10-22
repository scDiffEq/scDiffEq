
from .._model._supporting_functions._training._training_monitor import _TrainingMonitor

def _clear_training_monitor(TrainingMonitor, hyper_parameters, silent=False):

#     try:
#         TrainingMonitor
    return _TrainingMonitor(hyper_parameters.smoothing_momentum) # reinstantiates if exists
                                    
#     except:
#         pass
    #         print("No training monitor found...\n")
    #         return None