
from lightning import Callback
import pickle
import os


class IntermittentSaves(Callback):
    def __init__(self, frequency=50):
                
        self.frequency = frequency

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx, *args, **kwargs):
        
        """ """
                
        epoch = model.current_epoch
        
        if (epoch % self.frequency == 0) and (batch_idx == 0):
            dpath = os.path.join(trainer.logger.log_dir, "ckpt_outputs")
            if not os.path.exists(dpath):
                os.mkdir(dpath)

            fpath = os.path.join(dpath, "epoch_{}.batch_{}.pkl".format(epoch, batch_idx))
            pickle.dump(obj=outputs,
                        file=open(fpath, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL,
                       )
