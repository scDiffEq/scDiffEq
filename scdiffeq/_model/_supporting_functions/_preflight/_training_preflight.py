
# _training_preflight.py
__module_name__ = "_training_preflight.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# local imports #
# ------------- #
from ._choose_optimizer import _choose_optimizer
from ._choose_loss_function import _choose_loss_function


class TrainingParameters(object):
    def __init__(self):

        self.n_epochs = 1000
        self.train_proportion = 0.6
        self.valid_proportion = 0.2
        self.test_proportion = 0.2
        self.learning_rate = 1e-3
        self.validation_frequency = int(self.n_epochs / 20)
        self.visualization_frequency = int(self.n_epochs / 10)
        self.loss_function = _choose_loss_function("MSELoss")
        self.optimizer = _choose_optimizer(self, "RMSprop", self.learning_rate)
        self.smoothing_momentum = 0.99

    def update(self, parameter, value):

        if parameter == "loss_function":
            self.loss_function = _choose_loss_function(value)

        elif parameter == "optimizer":
            self.optimizer = _choose_optimizer(self, value, self.learning_rate)

        else:
            self.__setattr__(parameter, value)


def _preflight_parameters(ParamDict):
    
    training_params = TrainingParameters()
    
    for parameter, value in ParamDict.items():
        training_params.update(parameter, value)
    
    return training_params