import vinplots

def _axes_to_list(AxesDict):

    axes_list = []
    for i in AxesDict.keys():
        for j in AxesDict[i].keys():
            axes_list.append(AxesDict[i][j])

    return axes_list


def _build_training_loss_plot(nplots, ncols):

    """"""

    fig = vinplots.Plot()
    fig.construct(nplots=nplots, ncols=ncols, figsize=1, hspace=0.5)
    fig.modify_spines(
        ax="all",
        spines_to_delete=["top", "right"],
        spines_positioning_amount=15,
        spines_to_move=["bottom", "left"],
    )

    axes = _axes_to_list(fig.AxesDict)

    return fig, axes


def _plot_training_curve_single_ax(ax, training_loss_df, group):

    ax.fill_between(
        training_loss_df.index,
        training_loss_df["lower"],
        training_loss_df["upper"],
        alpha=0.5,
        color="lightblue",
        edgecolor="navy",
    )
    ax.plot(training_loss_df["mean"], color="navy", label=group, lw=3)
    ax.set_title("{} Layers {} Nodes".format(group[0], group[1]), fontsize=12)
    ax.set_xlabel("Epochs", fontsize=8)
    ax.set_ylabel("Wasserstein Distance", fontsize=8)


def _plot_training_loss_parameter_tuning(hp_table, results_dict, nplots, ncols):

    """"""

    fig, axes = _build_training_loss_plot(nplots, ncols)
    ax_count = 0
    for group, grouped_df in hp_table.groupby(["layers", "nodes"]):
        tuning_key = "layers_{}.nodes_{}".format(group[0], group[1])
        tuning_loss_df = results_dict[tuning_key]
        _plot_training_curve_single_ax(axes[ax_count], tuning_loss_df, group)
        ax_count += 1