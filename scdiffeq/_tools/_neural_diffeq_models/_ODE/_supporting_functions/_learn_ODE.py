import time
from torchdiffeq import odeint
from tqdm import tqdm_notebook as tqdm


def _training_preflight(
    adata,
    loss_func="MSE",
    learning_rate=1e-3,
    downsample_trajectory_factor=1,
    validation_frequency=20,
):

    adata.uns["loss_tracker"] = []
    adata.uns["epoch_training_loss_meter"] = []
    adata.uns["epoch_validation_loss_meter"] = []
    adata.uns["training_loss_meter"] = []
    adata.uns["validation_loss_meter"] = []
    adata.uns["test_loss_meter"] = []
    adata.uns["epoch_test_loss_meter"] = []
    adata.uns["validation_frequency"] = validation_frequency
    adata.uns["downsample_trajectory_factor"] = downsample_trajectory_factor
    adata.uns["time_meter"] = sdq.tl.RunningAverageMeter(0.97)
    adata.uns["loss_meter"] = sdq.tl.RunningAverageMeter(0.97)
    adata.uns["optimizer"] = optim.RMSprop(
        adata.uns["ODE"].parameters(), lr=learning_rate
    )

    if loss_func == "MSE":
        adata.uns["loss_func"] = torch.nn.MSELoss()
    else:
        print("Define a loss func.")


def _forward_integrate_single_trajectory(adata, trajectory):

    """


    Parameters:
    -----------
    adata

    trajectory

    Returns:
    --------
    predicted_trajectory

    """
    func = adata.uns["ODE"]
    y0 = torch.Tensor(adata.X[trajectory.index[0]])
    t = torch.Tensor(trajectory.time.values)
    predicted_trajectory = odeint(func=func, y0=y0, t=t)
    return predicted_trajectory


def _forward_integrate_one_training_epoch(adata, training_data_key="training"):

    """"""

    initial_time = time.time()
    loss_func = adata.uns["loss_func"]
    data = adata.uns["data_split_keys"][training_data_key]

    downsample_trajectory_factor = adata.uns["downsample_trajectory_factor"]
    epoch_loss = []

    for trajectory in data.obs.trajectory.unique():
        adata.uns["optimizer"].zero_grad()
        _trajectory = sdq.ut.isolate_trajectory(data, trajectory)[
            ::downsample_trajectory_factor
        ]
        true_trajectory = torch.Tensor(adata[_trajectory.index.astype(int)].X)
        predicted_trajectory = _forward_integrate_single_trajectory(adata, _trajectory)
        loss = loss_func(predicted_trajectory, true_trajectory)
        epoch_loss.append(loss.item())
        adata.uns["time_meter"].update(time.time() - initial_time)
        adata.uns["loss_meter"].update(loss.item())
        adata.uns["training_loss_meter"].append(loss.item())
        loss.backward()
        adata.uns["optimizer"].step()

    adata.uns["epoch_training_loss_meter"].append(np.array(epoch_loss).mean())


def _forward_integrate_one_batch_no_grad(adata, data_key):

    """For validation and evaluation forward integration."""

    initial_time = time.time()
    loss_func = adata.uns["loss_func"]
    data = adata.uns["data_split_keys"][data_key]

    downsample_trajectory_factor = adata.uns["downsample_trajectory_factor"]
    batch_loss = []

    for trajectory in data.obs.trajectory.unique():
        with torch.no_grad():
            _trajectory = sdq.ut.isolate_trajectory(data, trajectory)[
                ::downsample_trajectory_factor
            ]
            true_trajectory = torch.Tensor(adata[_trajectory.index.astype(int)].X)
            predicted_trajectory = _forward_integrate_single_trajectory(
                adata, _trajectory
            )
            loss = loss_func(predicted_trajectory, true_trajectory)
            batch_loss.append(loss.item())

            if data_key == "validation":
                adata.uns["validation_loss_meter"].append(loss.item())
            elif data_key == "test":
                adata.uns["test_loss_meter"].append(loss.item())

    if data_key == "validation":
        adata.uns["epoch_validation_loss_meter"].append(np.array(batch_loss).mean())
    if data_key == "test":
        adata.uns["epoch_test_loss_meter"].append(np.array(batch_loss).mean())


from IPython.display import clear_output


def _plot_training_loss(adata, show_only_epoch_loss=True, plot_validation=True):

    n_traj = adata.uns["data_split_keys"]["training"].obs.trajectory.nunique()
    iters = np.arange(len(adata.uns["epoch_training_loss_meter"]))
    clear_output(wait=True)
    if not show_only_epoch_loss:
        plt.plot(
            adata.uns["training_loss_meter"], alpha=0.25, label="Single Iteration Loss"
        )
        iters = iters * n_traj

    plt.plot(iters, adata.uns["epoch_training_loss_meter"], lw=3, label="Training")

    if plot_validation:
        val_iters = (
            np.arange(len(adata.uns["epoch_validation_loss_meter"]))
            * adata.uns["validation_frequency"]
        )
        val_loss = adata.uns["epoch_validation_loss_meter"]

        plt.plot(val_iters, val_loss, lw=3, label="Validation")
    plt.title("Mean Epoch Loss")
    plt.legend()
    plt.show()


def _count_datapoints_per_trajectory(adata):

    """"""

    traj_lengths = np.array([])

    for i in adata.obs.trajectory.unique():
        traj_lengths = np.append(
            traj_lengths, sdq.ut.isolate_trajectory(adata, i).shape[0]
        )

    mean_traj_length = traj_lengths.mean()
    print(mean_traj_length)


def _learn(
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
    #             _plot_training_loss(adata)

def learn(self, n_epochs, lr=1e-3, ds=1, val_freq=25):

    _learn(
        self.adata,
        n_epochs,
        learning_rate=lr,
        downsample_trajectory_factor=ds,
        validation_frequency=val_freq,
    )

def evaluate(self):

    """"""
    print("eval")
    # do nothing yet

def visualize(self):
    print("viz")

    # do nothing yet