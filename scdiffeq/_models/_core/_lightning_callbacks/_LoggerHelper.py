from pytorch_lightning import Callback
import torch
import time

class LoggerHelper(Callback):
    
    def on_train_epoch_end(self, trainer, model):
        model.log("train_epoch", torch.Tensor([model.current_epoch]).to(torch.float32))
        model.log("train_epoch_time", torch.Tensor([time.time()]).to(torch.float32))
        
    def on_validation_epoch_end(self, trainer, model):
        model.log("val_epoch", torch.Tensor([model.current_epoch]).to(torch.float32))
        model.log("val_epoch_time", torch.Tensor([time.time()]).to(torch.float32))