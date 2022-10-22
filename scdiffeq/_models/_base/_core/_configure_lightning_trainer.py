
__module_name__ = "_configure_lightning_trainer.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- : ----
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import os
import torch


from ._base_utility_functions import extract_func_kwargs


def configure_CSVLogger(
    model_save_dir: str = "scDiffEq_model",
    log_name: str = "lightning_logs",
    version=None,
    prefix="",
    flush_logs_every_n_steps=5,
):
    """
    model_save_dir
    log_name
    version
    prefix
    flush_logs_every_n_steps

    Notes:
    ------
    (1) Used as the default logger because it is the least complex and most predictable.
    (2) This function simply handle the args to pytorch_lighting.loggers.CSVLogger. While
        not functionally necessary, helps to clean up model code a bit.
    (3) doesn't change file names rather the logger name and what's added to the front of
        each log name event within the model.
    (4) Versioning contained / created automatically within by lightning logger
    """

    model_log_path = os.path.join(model_save_dir, log_name)

    logger = loggers.CSVLogger(
        save_dir=model_save_dir,
        name=log_name,
        version=version,
        prefix=prefix,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )

    return {"logger": logger, "log_path": model_log_path}


def accelerator():
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def configure_lightning_trainer(
    model_save_dir="scDiffEq_model",
    log_name="lightning_logs",
    version=None,
    prefix="",
    flush_logs_every_n_steps=5,
    max_epochs=1500,
    log_every_n_steps=1,
    reload_dataloaders_every_n_epochs=5,
    kwargs={},
):
    
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    logger_kwargs = extract_func_kwargs(func=loggers.CSVLogger, kwargs=locals())
    trainer_kwargs = extract_func_kwargs(func=Trainer, kwargs=locals())

    logging = configure_CSVLogger(**logger_kwargs)
    return Trainer(
        accelerator=accelerator(),
        devices=torch.cuda.device_count(),
        logger=logging["logger"],
        **trainer_kwargs
    )