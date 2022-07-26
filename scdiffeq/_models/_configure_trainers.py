from pytorch_lightning import Trainer

"""
This is a very rough / early implementation just to get the job done. Many convenience / preset adjustments
are yet to be made! 
"""


class LightningTrainers:
    def __init__(
        self,
        pretraining_epochs=500,
        training_epochs=2500,
        gpus=0,
        enable_model_summary=False,
        log_every_n_steps=5,
    ):

        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.gpus = gpus
        self.enable_model_summary = enable_model_summary
        self.log_every_n_steps = log_every_n_steps

    def setup_pretrain(self, **kwargs):

        self.pretraining = Trainer(
            max_epochs=self.training_epochs,
            gpus=self.gpus,
            enable_model_summary=self.enable_model_summary,
            log_every_n_steps=self.log_every_n_steps,
            **kwargs,
        )

    def setup_train(self, **kwargs):

        self.training = Trainer(
            max_epochs=self.pretraining_epochs,
            gpus=self.gpus,
            enable_model_summary=self.enable_model_summary,
            log_every_n_steps=self.log_every_n_steps,
            **kwargs,
        )


def _configure_lightning_trainers(
    pretrain=True, train=True, pretrainer_kwargs={}, trainer_kwargs={}
):

    Trainers = LightningTrainers()
    if pretrain:
        Trainers.setup_pretrain(**pretrainer_kwargs)
    if train:
        Trainers.setup_train(**trainer_kwargs)

    return Trainers
