
from ...utils import AutoParseBase


# -- Main module class: ----------------------------------------------------------------
class LossLogger(AutoParseBase):
    
    BackPropDict = {}
    
    def __init__(self, backprop_losses=[], unlogged_stages=["predict", "test"]):
        self.__parse__(locals())

    def __call__(self, model, LossDict, stage):
        
        for loss_key, loss_vals in LossDict.items():
            if loss_key in self.backprop_losses:
                self.BackPropDict[loss_key] = loss_vals
            if loss_vals.dim() == 0:
                log_msg = "{}_{}_{}".format(stage, 0, loss_key)
                model.log(log_msg, loss_vals)
            else:
                for i, loss_val in enumerate(loss_vals):
                    log_msg = "{}_{}_{}".format(stage, i, loss_key)
                    model.log(log_msg, loss_val)
                    
        return self.BackPropDict