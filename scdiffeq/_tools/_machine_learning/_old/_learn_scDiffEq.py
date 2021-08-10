
def _learn_scDiffEq(
    adata,
    n_epochs,
    learning_rate=1e-3,
    downsample_trajectory_factor=1,
    validation_frequency=25,
):

    """"""

    _training_preflight(
        adata,
        loss_func="MSE",
        learning_rate=learning_rate,
        downsample_trajectory_factor=downsample_trajectory_factor,
        validation_frequency=validation_frequency,
    )

    for epoch in tqdm(range(n_epochs)):
        _forward_integrate_one_training_epoch(adata, training_data_key="training")
        if epoch % adata.uns["validation_frequency"] == 0:
            _forward_integrate_one_batch_no_grad(adata, data_key="validation")