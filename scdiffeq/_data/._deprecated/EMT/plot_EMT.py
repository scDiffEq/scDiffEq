import matplotlib.pyplot as plt


def subplot_presets(
    ax, x, y, xlab, ylab, title, color, label, x_line=None, stability=None
):

    ax.set_xlabel(xlab, fontsize=15)
    ax.set_ylabel(ylab, fontsize=15)
    ax.set_title(title, fontsize=20, y=1.05)
    ax.scatter(x, y, s=4, c=color, label=label)
    #     ax.plot(x, y, marker="o", linewidth=2, markersize=4, color=color)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if stability != None:
        label = stability
    else:
        label = None

    ax.axvline(
        x_line, linewidth=2.5, color="r", alpha=0.5, linestyle="dashed", label=label
    )
    #     if x_line != None:

    #         ax.plot(x=np.full((range(len(y))), 300), y=range(y))
    return ax


def plot_EMH(
    reaching_epithelial, reaching_mesenchymal, reaching_hybrid, mode="ZEB/miR200"
):

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    epi, mes, hyb = axes[0], axes[1], axes[2]

    if mode == "SNAIL/miR34":

        epi_pl1_value, mes_pl1_value, hyb_pl1_value = (
            reaching_epithelial.miR34,
            reaching_mesenchymal.miR34,
            reaching_hybrid.miR34,
        )
        epi_pl2_value, mes_pl2_value, hyb_pl2_value = (
            reaching_epithelial.mSNAIL,
            reaching_mesenchymal.mSNAIL,
            reaching_hybrid.mSNAIL,
        )

        miR_label = "miR34"
        mRNA_label = "SNAIL"

    elif mode == "ZEB/miR200":

        epi_pl1_value, mes_pl1_value, hyb_pl1_value = (
            reaching_epithelial.miR200,
            reaching_mesenchymal.miR200,
            reaching_hybrid.miR200,
        )
        epi_pl2_value, mes_pl2_value, hyb_pl2_value = (
            reaching_epithelial.mZEB,
            reaching_mesenchymal.mZEB,
            reaching_hybrid.mZEB,
        )

        miR_label = "miR200"
        mRNA_label = "ZEB"

    subplot_presets(
        epi,
        reaching_epithelial.time.values,
        epi_pl1_value / 30000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Epithelial",
        color="orange",
        label=miR_label,
        x_line=300,
    )

    subplot_presets(
        epi,
        reaching_epithelial.time.values,
        epi_pl2_value / 1000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Epithelial",
        color="navy",
        label=mRNA_label,
        x_line=300,
    )

    subplot_presets(
        mes,
        reaching_mesenchymal.time.values,
        mes_pl1_value / 30000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Mesenchymal",
        color="orange",
        label=miR_label,
        x_line=300,
    )
    subplot_presets(
        mes,
        reaching_mesenchymal.time.values,
        mes_pl2_value / 1000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Mesenchymal",
        color="navy",
        label=mRNA_label,
        x_line=300,
    )

    subplot_presets(
        hyb,
        reaching_hybrid.time.values,
        hyb_pl1_value / 30000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Hybrid",
        color="orange",
        label=miR_label,
        x_line=300,
    )
    subplot_presets(
        hyb,
        reaching_hybrid.time.values,
        hyb_pl2_value / 1000,
        xlab="Time (h)",
        ylab="Molecules",
        title="Hybrid",
        color="navy",
        label=mRNA_label,
        x_line=300,
        stability="Stable state",
    )

    plt.legend(
        markerscale=3,
        edgecolor="w",
        fontsize=14,
        handletextpad=None,
        bbox_to_anchor=(0.5, 0.0, 0.70, 1),
    )
