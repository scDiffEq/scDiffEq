
from ._clear_training_monitor import _clear_training_monitor
from ._reset_network_model import _reset_network_model


def _reset_scDiffEq(network_model, TrainingMonitor, hyper_parameters, silent):
    
    network_model = _reset_network_model(network_model)
    TrainingMonitor = _clear_training_monitor(TrainingMonitor, hyper_parameters, silent)
    
    return network_model, TrainingMonitor