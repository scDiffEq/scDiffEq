import os
import pickle
import torch
import vintools as v


def _write_pickle_dict(adata, path, uns_key, verbose=False):
    """"""

    if verbose:
        print("Using pickle to save adata.uns[{}]...".format(uns_key))

    f_path = os.path.join(path, (uns_key + ".pkl"))
    pickle.dump(adata.uns[uns_key], open(uns_key, "wb"))


def _save_uns_tensors(adata, path, uns_key):

    """"""

    f_path = os.path.join(path, (uns_key + ".pt"))
    torch.save(obj=adata.uns[uns_key], f=f_path)


def _save_torch_model(self, path, tostring=False):

    """
    adata
        anndata._core.anndata.AnnData

    epoch
        current epoch
        type: int
    
    path
        type: str
    """

    epoch = self.epoch
    adata = self.adata

    model = adata.uns["ODE"]
    optimizer = adata.uns["optimizer"]

    if tostring:
        adata.uns["optimizer"] = str(adata.uns["optimizer"])
        adata.uns["ODE"] = str(adata.uns["ODE"])

    latest_train_loss = adata.uns["loss"]["train_loss"][-1]
    latest_valid_loss = adata.uns["loss"]["valid_loss"][-1]
    if type(model) is not str:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": latest_train_loss,
                "valid_loss": latest_valid_loss,
            },
            path,
        )


def _save_adata_uns(
    self,
    pickle_dump_list=["pca", "loss"],
    pass_keys=["split_data", "data_split_keys", "RunningAverageMeter"],
):

    """
    
    Notes:
    ------
    (1) if uns_key is in the predefined list of keys to pass or is an int, pass. these will be overlooked / deleted.
    """
    model_checkpoint_path = os.path.join(self._outs_path, "model_checkpoints")
    v.ut.mkdir_flex(model_checkpoint_path)

    self.backup_uns_dict = {}

    for uns_key in self.adata.uns_keys():
        self.backup_uns_dict[uns_key] = self.adata.uns[uns_key]

        ###### pass ######
        if uns_key in pass_keys:
            pass
        elif self.adata.uns[uns_key].__class__.__name__ == "int":
            pass
        ###### pass ######

        ###### save loss func ######
        elif uns_key == "loss_func":
            self.adata.uns["loss_func"] = str(self.adata.uns["loss_func"])
        ###### save loss func ######

        ###### write dicts ######
        elif uns_key in pickle_dump_list:
            _write_pickle_dict(adata=self.adata, path=self._uns_path, uns_key=uns_key)
        ###### write dicts ######

        ###### write tensors ######
        elif self.adata.uns[uns_key].__class__.__name__ == "Tensor":
            _save_uns_tensors(adata=self.adata, path=self._uns_path, uns_key=uns_key)
        ###### write tensors ######

        ###### save model ######
        elif uns_key == "ODE" or "optimizer":
            path = os.path.join(model_checkpoint_path, "model_{}".format(self.epoch))
            _save_torch_model(self, path=path, tostring=True)
        ###### save model ######
        else:
            print("Undealt with: {}".format(uns_key))
            #     save_uns_dict[uns_key] = adata.uns[uns_key]
    #         print(adata.uns[uns_key].__class__.__name__)
    del self.adata.uns
