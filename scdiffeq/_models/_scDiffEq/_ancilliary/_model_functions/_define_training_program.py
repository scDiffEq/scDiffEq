
def _define_training_program(
    epochs, lr_schedule, validation_frequency, checkpoint_frequency, notebook
):

    return {
        "epochs": epochs,
        "lr_schedule": lr_schedule,
        "validation_frequency": validation_frequency,
        "checkpoint_frequency": checkpoint_frequency,
        "notebook": notebook,
    }