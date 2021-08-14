
import os
import vintools as v

from ._forward_integration_functions._forward_integrate_epoch import _forward_integrate_epoch
from ._plot_neural_ODE_training import _plot_loss
from ._save_adata_uns import _save_torch_model
from .._machine_learning._save import _save

def _learn_neural_ODE(
    self,
    n_epochs=100, 
    plot_progress=True, 
    plot_summary=True, 
    notebook=False, 
    smoothing_factor=3, 
    visualization_frequency=5,
    save_frequency=5,
    save_path=False
):

    """"""             
    
    model_checkpoint_path = os.path.join(self._outs_path, "model_checkpoints")
    v.ut.mkdir_flex(model_checkpoint_path)
    
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        bar_format = None
    else:
        from tqdm import tqdm as tqdm
        bar_format = '{l_bar}{bar:60}{r_bar}{bar:-60b}'
        
    for epoch in tqdm((range(1, n_epochs + 1)), desc="Training progress", bar_format=bar_format):
#         self.adata_copy = adata
        self.epoch = self.adata.uns['last_epoch'] = epoch
        _forward_integrate_epoch(self, epoch)
        if plot_progress:
            if epoch % self.visualization_frequency == 0:
                test_predict_path = self._imgs_path + "epoch_{}_training_progress.png".format(self.epoch)
                _plot_loss(self.adata, groupsize=smoothing_factor, save_path=test_predict_path)
                torch_model_path = os.path.join(model_checkpoint_path, "model_{}".format(self.epoch))
                _save_torch_model(self, torch_model_path)
                
#         if epoch % save_frequency == 0:
#             _save(self, 
#                   outdir=self.outdir, 
#                   pickle_dump_list = ["pca", "loss"], 
#                   pass_keys = ["split_data", "data_split_keys", "RunningAverageMeter"],
#                   put_back=True, 
#                  )
#             adata = self.adata_copy

    if plot_summary:
        _plot_loss(self.adata, groupsize=smoothing_factor, save_path=save_path)
        
#         """"""
#         try:
#             self.epoch = self.adata.uns['last_epoch']
#         except:
#             self.epoch = self.epoch
#         _save(
#             self,
#             outdir=outdir,
#             pickle_dump_list=pickle_dump_list,
#             pass_keys=pass_keys
#         )