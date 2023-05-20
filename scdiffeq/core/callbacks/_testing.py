
from lightning import Callback
from datetime import datetime
import pickle
import os

class Testing(Callback):
    def __init__(self):
        self.now = str(datetime.now())

    def _mk_output_dir(self, trainer):
        self.dpath = os.path.join(trainer.logger.log_dir, "model_test_outputs")
        if not os.path.exists(self.dpath):
            os.mkdir(self.dpath)
            
    def on_test_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx, *args, **kwargs):
        
        """ """
        
        self._mk_output_dir(trainer)
        
        fpath = os.path.join(self.dpath, "test_outputs.{}.pkl".format(self.now))
        pickle.dump(obj=outputs,
                    file=open(fpath, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                   )
