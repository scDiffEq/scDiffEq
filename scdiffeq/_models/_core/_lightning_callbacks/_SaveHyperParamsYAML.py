import os
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml


class SaveHyperParamsYAML(Callback):
    def on_train_start(self, trainer, model):
        path = os.path.join(model.logger.log_dir, "user_hparams.yaml")
        save_hparams_to_yaml(config_yaml=path, hparams=model.hparam_dict)