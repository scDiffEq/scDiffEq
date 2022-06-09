
__module_name__ = "_define_training_program.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


def _define_training_program(
    epochs,
    pretrain_VAE_epochs,
    lr_schedule,
    validation_frequency,
    checkpoint_frequency,
    use_lineages,
    use_key,
    time_key,
    notebook,
):

    return {
        "epochs": epochs,
        "pretrain_VAE_epochs":pretrain_VAE_epochs,
        "lr_schedule": lr_schedule,
        "validation_frequency": validation_frequency,
        "checkpoint_frequency": checkpoint_frequency,
        "use_lineages": use_lineages,
        "use_key": use_key,
        "time_key": time_key,
        "notebook": notebook,
    }