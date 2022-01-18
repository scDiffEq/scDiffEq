
# _training_preflight.py
__module_name__ = "_training_preflight.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


import licorice

# local imports #
# ------------- #
from ._choose_optimizer import _choose_optimizer
from ._choose_loss_function import _choose_loss_function

class Training_HyperParameters(object):
    def __init__(self, network_model):

        self.n_epochs = 1000
        self.train_proportion = 0.6
        self.valid_proportion = 0.2
        self.test_proportion = 0.2
        self.learning_rate = 1e-3
        self.validation_frequency = int(self.n_epochs / 20)
        self.visualization_frequency = int(self.n_epochs / 10)
        self.loss_function = _choose_loss_function("MSELoss")
        self.optimizer = _choose_optimizer(network_model, "RMSprop", self.learning_rate)
        self.smoothing_momentum = 0.99
        self.learn_by = "trajectory"

    def _update_parameter(self, parameter, value):

        if parameter == "loss_function":
            self.loss_function = _choose_loss_function(value)

        elif parameter == "optimizer":
            self.optimizer = _choose_optimizer(self, value, self.learning_rate)

        else:
            self.__setattr__(parameter, value)
    
    def update(self, ParamDict):
        
        """
        Update all passed parameters in a ParamDict. 
        """
        
        for parameter, value in ParamDict.items():
            self._update_parameter(parameter, value)
            message1 = licorice.font_format("Parameter adjusted: ", ["BOLD", "RED"])
            message2 = licorice.font_format("{} = {}\n".format(parameter, value), ["BOLD"])
            print(message1, message2)
        
class UX_Preferences(object):
    
    def __init__(self):
        
        self.silent=False
        self.plot_title_fontsize=9
        self.plot_ticks_fontsize=6
        self.plot_axis_label_fontsize=8
        self.validation_plot_savepath = "/imgs/validation/"
        self.evaluation_plot_savepath = "/imgs/evaluation/"
        
    def update(self, PreferencesDict):
        
        """
        Update all pass preferences in a PreferencesDict.
        """
        
        for preference, value in PreferencesDict.items():
            self.preference = value
            message1 = licorice.font_format("Preference added: ", ["BOLD", "CYAN"])
            message2 = licorice.font_format("{} = {}\n".format(preference, value), ["BOLD"])
            print(message1, message2)
                          
def _catch_and_sort_preflight_kwargs(training_params, ux_preferences, **kwargs):

    training_params_keys = training_params.__dict__.keys()
    ux_preferences_keys = ux_preferences.__dict__.keys()

    _InvalidKwargs = {}

    ParamUpdateDict, PreferencesUpdateDict = {}, {}

    for key, value in kwargs.items():
        if key in training_params_keys:
            ParamUpdateDict[key] = value
        elif key in ux_preferences_keys:
            PreferencesUpdateDict[key] = value
        else:
            message = licorice.font_format("Keyword: {} not found.".format(key), ["BOLD", "RED"])
            print(
                "{}\nKey-value argument pair stored as: {} : {} in DiffEq._InvalidKwargs.".format(
                    message, key, value
                )
            )
            _InvalidKwargs[key] = value

    return ParamUpdateDict, PreferencesUpdateDict, _InvalidKwargs
                
def _preflight_parameters(network_model, **kwargs):
    
    training_params = Training_HyperParameters(network_model)
    ux_preferences = UX_Preferences()
    
    ParamUpdateDict, PreferencesUpdateDict, _InvalidKwargs = _catch_and_sort_preflight_kwargs(training_params, 
                                                                                              ux_preferences, 
                                                                                              **kwargs)
    training_params.update(ParamUpdateDict)
    ux_preferences.update(PreferencesUpdateDict)
    
#     if ParamUpdateDict:
#         training_params.update(ParamUpdateDict)
    
#     if PreferencesUpdateDict:
#         ux_preferences.update(PreferencesUpdateDict)
                  
    return training_params, ux_preferences, _InvalidKwargs


        
        