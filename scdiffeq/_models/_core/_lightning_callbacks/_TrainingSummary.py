
from pytorch_lightning import Callback
import time
import os    
    

class TrainingSummary(Callback):
    def on_train_start(self, trainer, model):

        self.train_start = time.time()
        path = os.path.join(model.logger.log_dir, "training_summary.txt")
        f = open(path, "w")
        f.write("{}\t{}\n".format("train_start", self.train_start))
        f.close()
        
    def on_train_end(self, trainer, model):
        
        self.train_end = time.time()
        path = os.path.join(model.logger.log_dir, "training_summary.txt")
        
        best_model_path = trainer.callbacks[-1].best_model_path
        best_model_score = trainer.callbacks[-1].best_model_score
        
        f = open(path, "a")
        f.write("{}\t{}\n".format("train_end", self.train_end))
        f.write("{}\t{}\n".format("best_model_ckpt", best_model_path))
        f.write("{}\t{}\n".format("best_model_score", best_model_score))
        f.close()